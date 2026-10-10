"""Curva di sopravvivenza vs spinta, su GPU/MJX, per checkpoint addestrati con train_gpu (o CPU).
Stesso protocollo di push_curve.py (spinta fissa ogni 1.5-3 s, 10 s, sopravvivenza = time-out) ma con molti piu' episodi.
Uso: python -m spiking_rl.eval.push_curve_gpu runs/X [--envs 512] [--mags 0.5 1 1.5]"""
import argparse
import json
import os

import numpy as np
import torch

from spiking_rl.envs.mjx_go2 import Go2BalanceMJX
from spiking_rl.eval.push_curve import load


def evaluate(run_dir, mags, n_envs, seed=50_000, **env_kw):
    agent, ck, c = load(run_dir)
    agent = agent.cuda()
    mean = torch.as_tensor(ck["norm_mean"], dtype=torch.float32, device="cuda")
    std = torch.sqrt(torch.as_tensor(ck["norm_var"], dtype=torch.float32, device="cuda") + 1e-8)
    out = {}
    for m in mags:
        env = Go2BalanceMJX(n_envs, seed=seed, push_vel=(m, m), push_every=(1.5, 3.0), **env_kw)
        obs = env.reset()
        alive = torch.ones(n_envs, dtype=torch.bool, device="cuda")
        surv = torch.zeros(n_envs, dtype=torch.bool, device="cuda")
        for _ in range(env.max_steps):
            with torch.no_grad():
                a, _, _ = agent.act(((obs - mean) / std).clamp(-10, 10), deterministic=True)
            obs, _, _, term, trunc, _, _ = env.step(a)
            surv |= alive & trunc & ~term
            alive &= ~(term | trunc)
        s = float(surv.float().mean())
        out[str(m)] = {"survival": s, "n": n_envs}
        print(f"push {m:.1f} m/s: sopravvivenza {s * 100:.1f}% ({int(surv.sum())}/{n_envs})", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--mags", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    a = p.parse_args()
    res = evaluate(a.run_dir, a.mags, a.envs)
    json.dump(res, open(os.path.join(a.run_dir, "push_curve_gpu.json"), "w"), indent=1)
