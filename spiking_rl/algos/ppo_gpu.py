"""PPO su GPU con env MJX batched (Go2BalanceMJX). Stessa logica di `ppo.py` (normalizzazione di obs e reward in stile
Gymnasium, bootstrap sui time-limit, actor ANN/SNN intercambiabile) ma tutto su tensori CUDA, senza passare dalla CPU.
Il checkpoint e' compatibile con `eval/push_curve.py` (valutazione sulla versione CPU = test sim2sim)."""
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

from spiking_rl.algos.ppo import PPOConfig
from spiking_rl.envs.mjx_go2 import Go2BalanceMJX
from spiking_rl.models import ActorCritic


class TorchNorm:
    """Media/varianza correnti (algoritmo parallelo di Chan), su GPU."""

    def __init__(self, shape, device, eps=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = eps

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + d * bc / tot
        m2 = self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot
        self.var, self.count = m2 / tot, tot

    def __call__(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)


def save_checkpoint(path, agent, norm, cfg, n_obs, n_act):
    torch.save({"state_dict": {k: v.cpu() for k, v in agent.state_dict().items()},
                "norm_mean": norm.mean.cpu().numpy().astype(np.float64),
                "norm_var": norm.var.cpu().numpy().astype(np.float64),
                "n_obs": n_obs, "n_act": n_act, "discrete": False,
                "cfg": {k: v for k, v in vars(cfg).items() if k != "log"}}, path)


def train_gpu(cfg: PPOConfig):
    dev = "cuda"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env = Go2BalanceMJX(cfg.num_envs, seed=cfg.seed, **cfg.env_kwargs)
    n_obs, n_act, N, R = env.obs_dim, env.act_dim, cfg.num_envs, cfg.rollout
    agent = ActorCritic(n_obs, n_act, False, cfg.actor, cfg.hidden, T=cfg.T, neuron=cfg.neuron,
                        encoding=cfg.encoding, init_log_std=cfg.init_log_std).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    norm, rnorm = TorchNorm((n_obs,), dev), TorchNorm((), dev)
    acc_ret = torch.zeros(N, device=dev)
    n_iters = cfg.total_steps // (N * R)
    os.makedirs(cfg.out_dir, exist_ok=True)
    f = open(os.path.join(cfg.out_dir, "progress.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["step", "ep_return", "rate_l1", "rate_l2", "sps"])
    obs_raw = env.reset()
    norm.update(obs_raw)
    t0, step = time.time(), 0
    last_ret = float("nan")
    rates = [0.0, 0.0]

    for it in range(n_iters):
        frac = it / max(1, n_iters)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * (1 - frac)
        if cfg.actor == "snn":
            agent.actor.set_alpha(cfg.alpha_start + (cfg.alpha_end - cfg.alpha_start) * min(1.0, frac / 0.3))
        b_obs = torch.zeros(R, N, n_obs, device=dev)
        b_act = torch.zeros(R, N, n_act, device=dev)
        b_lp, b_rew, b_done, b_val = (torch.zeros(R, N, device=dev) for _ in range(4))
        ep_rets = []
        for t in range(R):
            obs = norm(obs_raw)
            with torch.no_grad():
                a, lp, _ = agent.act(obs)
                v = agent.value(obs)
            nobs_raw, fobs, r, term, trunc, ep_ret, _ = env.step(a)
            done = term | trunc
            # normalizzazione del reward (come gym.wrappers.NormalizeReward): scala per la std del ritorno scontato
            acc_ret = acc_ret * cfg.gamma * (~done).float() + r
            rnorm.update(acc_ret)
            r = r / torch.sqrt(rnorm.var + 1e-8)
            tr_only = trunc & ~term
            if bool(tr_only.any()):
                with torch.no_grad():
                    fv = agent.value(norm(fobs))
                r = r + cfg.gamma * fv * tr_only.float()
            b_obs[t], b_act[t], b_lp[t], b_val[t] = obs, a, lp, v
            b_rew[t], b_done[t] = r, done.float()
            if bool(done.any()):
                ep_rets.append(ep_ret[done])
            obs_raw = nobs_raw
            norm.update(obs_raw)
            step += N
        with torch.no_grad():
            nv = agent.value(norm(obs_raw))
        adv = torch.zeros(R, N, device=dev)
        last = 0
        for t in reversed(range(R)):
            nxt = nv if t == R - 1 else b_val[t + 1]
            nd = 1.0 - b_done[t]
            delta = b_rew[t] + cfg.gamma * nxt * nd - b_val[t]
            last = delta + cfg.gamma * cfg.lam * nd * last
            adv[t] = last
        ret = adv + b_val
        fo, fa, flp = b_obs.reshape(-1, n_obs), b_act.reshape(-1, n_act), b_lp.reshape(-1)
        fadv, fret = adv.reshape(-1), ret.reshape(-1)
        B = fo.shape[0]
        mb = B // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = torch.randperm(B, device=dev)
            for s in range(0, B - mb + 1, mb):
                ix = perm[s:s + mb]
                _, nlp, ent = agent.act(fo[ix], fa[ix])
                ratio = (nlp - flp[ix]).exp()
                a_ = fadv[ix]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = torch.max(-a_ * ratio, -a_ * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                vl = 0.5 * ((agent.value(fo[ix]) - fret[ix]) ** 2).mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()
        if cfg.actor == "snn":
            rates = agent.actor.last_rates
        if ep_rets:
            last_ret = float(torch.cat(ep_rets).mean())
        w.writerow([step, last_ret, rates[0], rates[1] if len(rates) > 1 else 0.0, int(step / (time.time() - t0))])
        f.flush()
        if it % 25 == 0 or it == n_iters - 1:
            print(f"it {it}/{n_iters} step {step:,} ep_return {last_ret:.2f} sps {int(step / (time.time() - t0)):,}", flush=True)
            save_checkpoint(os.path.join(cfg.out_dir, "agent.pt"), agent, norm, cfg, n_obs, n_act)
    f.close()
    save_checkpoint(os.path.join(cfg.out_dir, "agent.pt"), agent, norm, cfg, n_obs, n_act)
    return {"train_last_return": last_ret, "wall_s": time.time() - t0, "steps": step}
