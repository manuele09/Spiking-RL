"""PPO di riferimento (stile CleanRL) con actor intercambiabile ANN/SNN. Solo CPU, env Gymnasium vettorizzati."""
import csv
import os
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import spiking_rl.envs  # noqa: F401  (registra Go2Balance-v0)
from spiking_rl.models import ActorCritic


@dataclass
class PPOConfig:
    env_id: str = "Pendulum-v1"
    actor: str = "ann"  # ann | snn
    T: int = 4
    neuron: str = "lif"
    encoding: str = "direct"
    hidden: tuple = (128, 128)
    seed: int = 0
    total_steps: int = 200_000
    num_envs: int = 8
    rollout: int = 256
    epochs: int = 10
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    alpha_start: float = 0.5  # pendenza surrogate (scheduling lineare nel primo 30%)
    alpha_end: float = 2.0
    out_dir: str = "runs/tmp"
    eval_episodes: int = 20
    async_envs: bool = False
    env_kwargs: dict = field(default_factory=dict)
    log: dict = field(default_factory=dict)


class RunningNorm:
    def __init__(self, shape, eps=1e-4):
        self.mean, self.var, self.count = np.zeros(shape), np.ones(shape), eps

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + d * bc / tot
        m2 = self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot
        self.var, self.count = m2 / tot, tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


def make_env(env_id, gamma, normalize_reward, env_kwargs=None):
    def thunk():
        e = gym.make(env_id, **(env_kwargs or {}))
        e = gym.wrappers.RecordEpisodeStatistics(e)
        if normalize_reward and isinstance(e.action_space, gym.spaces.Box):
            e = gym.wrappers.NormalizeReward(e, gamma=gamma)
        return e
    return thunk


def train(cfg: PPOConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    VecEnv = gym.vector.AsyncVectorEnv if cfg.async_envs else gym.vector.SyncVectorEnv
    envs = VecEnv([make_env(cfg.env_id, cfg.gamma, True, cfg.env_kwargs) for _ in range(cfg.num_envs)],
                  autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    discrete = isinstance(envs.single_action_space, gym.spaces.Discrete)
    n_obs = int(np.prod(envs.single_observation_space.shape))
    n_act = int(envs.single_action_space.n) if discrete else int(np.prod(envs.single_action_space.shape))
    agent = ActorCritic(n_obs, n_act, discrete, cfg.actor, cfg.hidden, T=cfg.T, neuron=cfg.neuron,
                        encoding=cfg.encoding)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    norm = RunningNorm((n_obs,))
    low = None if discrete else envs.single_action_space.low
    high = None if discrete else envs.single_action_space.high

    def to_env(a):
        return a.numpy() if discrete else np.clip(a.numpy(), low, high)

    obs_raw, _ = envs.reset(seed=cfg.seed)
    norm.update(obs_raw)
    N, R = cfg.num_envs, cfg.rollout
    n_iters = cfg.total_steps // (N * R)
    os.makedirs(cfg.out_dir, exist_ok=True)
    f = open(os.path.join(cfg.out_dir, "progress.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["step", "ep_return", "rate_l1", "rate_l2", "sps"])
    recent, t0, step = [], time.time(), 0
    rates = [0.0, 0.0]

    for it in range(n_iters):
        frac = it / max(1, n_iters)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * (1 - frac)
        if cfg.actor == "snn":
            agent.actor.set_alpha(cfg.alpha_start + (cfg.alpha_end - cfg.alpha_start) * min(1.0, frac / 0.3))
        b_obs = torch.zeros(R, N, n_obs)
        b_act = torch.zeros((R, N) if discrete else (R, N, n_act))
        b_lp, b_rew, b_done, b_val = (torch.zeros(R, N) for _ in range(4))
        for t in range(R):
            obs = torch.as_tensor(norm(obs_raw), dtype=torch.float32)
            with torch.no_grad():
                a, lp, _ = agent.act(obs)
                v = agent.value(obs)
            nobs_raw, r, term, trunc, info = envs.step(to_env(a))
            r = torch.as_tensor(r, dtype=torch.float32)
            # bootstrap sui time-limit: SAME_STEP restituisce l'obs di reset, la finale e' in info
            if trunc.any():
                idx = np.where(trunc & ~term)[0]
                if len(idx):
                    fin = torch.as_tensor(norm(np.stack([info["final_obs"][i] for i in idx])), dtype=torch.float32)
                    with torch.no_grad():
                        r[idx] += cfg.gamma * agent.value(fin)
            done = term | trunc
            b_obs[t], b_act[t], b_lp[t], b_val[t] = obs, a, lp, v
            b_rew[t], b_done[t] = r, torch.as_tensor(done, dtype=torch.float32)
            if "final_info" in info and "episode" in info["final_info"]:
                ep = info["final_info"]["episode"]
                recent.extend(float(x) for x in ep["r"][ep["_r"]])
            obs_raw = nobs_raw
            norm.update(obs_raw)
            step += N
        with torch.no_grad():
            nv = agent.value(torch.as_tensor(norm(obs_raw), dtype=torch.float32))
        adv = torch.zeros(R, N)
        last = 0
        for t in reversed(range(R)):
            nxt = nv if t == R - 1 else b_val[t + 1]
            nd = 1.0 - b_done[t]
            delta = b_rew[t] + cfg.gamma * nxt * nd - b_val[t]
            last = delta + cfg.gamma * cfg.lam * nd * last
            adv[t] = last
        ret = adv + b_val
        fo, fa, flp = b_obs.reshape(-1, n_obs), b_act.reshape((-1,) + b_act.shape[2:]), b_lp.reshape(-1)
        fadv, fret = adv.reshape(-1), ret.reshape(-1)
        B = fo.shape[0]
        mb = B // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = torch.randperm(B)
            for s in range(0, B, mb):
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
        w.writerow([step, np.mean(recent[-20:]) if recent else float("nan"),
                    rates[0], rates[1] if len(rates) > 1 else 0.0, int(step / (time.time() - t0))])
        f.flush()
    f.close()

    # valutazione deterministica con normalizzazione congelata
    env = gym.make(cfg.env_id, **cfg.env_kwargs)
    rets = []
    for ep in range(cfg.eval_episodes):
        o, _ = env.reset(seed=10_000 + ep)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                a, _, _ = agent.act(torch.as_tensor(norm(o[None]), dtype=torch.float32), deterministic=True)
            o, r, te, tr, _ = env.step(a[0].numpy() if discrete else np.clip(a[0].numpy(), low, high))
            total += r
            done = te or tr
        rets.append(total)
    envs.close()
    torch.save({"state_dict": agent.state_dict(), "norm_mean": norm.mean, "norm_var": norm.var,
                "n_obs": n_obs, "n_act": n_act, "discrete": discrete,
                "cfg": {k: v for k, v in vars(cfg).items() if k != "log"}},
               os.path.join(cfg.out_dir, "agent.pt"))
    return {"eval_mean": float(np.mean(rets)), "eval_std": float(np.std(rets)),
            "train_last20": float(np.mean(recent[-20:])) if recent else float("nan")}
