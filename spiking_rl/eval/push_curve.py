"""Curva di sopravvivenza vs intensita' della spinta (L1). Uso:
python -m spiking_rl.eval.push_curve runs/go2_l1_ann_s0 [--episodes 30] [--mags 0.5 1 1.5 2 2.5]
Spinte a intensita' fissa ogni 1.5-3 s, episodi di 10 s; sopravvivenza = episodio arrivato al time-out.
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

import spiking_rl.envs  # noqa: F401
from spiking_rl.models import ActorCritic


def load(run_dir):
    ck = torch.load(os.path.join(run_dir, "agent.pt"), weights_only=False)
    c = ck["cfg"]
    agent = ActorCritic(ck["n_obs"], ck["n_act"], ck["discrete"], c["actor"], tuple(c["hidden"]), T=c["T"],
                        neuron=c["neuron"], encoding=c["encoding"])
    agent.load_state_dict(ck["state_dict"])
    agent.eval()
    return agent, ck, c


def evaluate(run_dir, mags, episodes, env_id="Go2Balance-v0"):
    agent, ck, c = load(run_dir)
    out = {}
    for m in mags:
        surv, rets, tilt, hdev = 0, [], [], []
        env = gym.make(env_id, push_vel=(m, m), push_every=(1.5, 3.0))
        for ep in range(episodes):
            o, _ = env.reset(seed=50_000 + ep)
            tot, mt = 0.0, 0.0
            while True:
                x = torch.as_tensor(np.clip((o[None] - ck["norm_mean"]) / np.sqrt(ck["norm_var"] + 1e-8), -10, 10),
                                    dtype=torch.float32)
                with torch.no_grad():
                    a, _, _ = agent.act(x, deterministic=True)
                o, r, te, tr, info = env.step(a[0].numpy())
                tot += r
                mt = max(mt, info["tilt"])
                if te or tr:
                    break
            surv += int(tr and not te)
            rets.append(tot)
            tilt.append(mt)
        out[str(m)] = {"survival": surv / episodes, "return": float(np.mean(rets)), "max_tilt_rad": float(np.mean(tilt))}
        print(f"push {m:.1f} m/s: sopravvivenza {surv}/{episodes}  ritorno {np.mean(rets):.2f}  inclinazione max {np.mean(tilt):.2f} rad")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--mags", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5])
    a = p.parse_args()
    res = evaluate(a.run_dir, a.mags, a.episodes)
    json.dump(res, open(os.path.join(a.run_dir, "push_curve.json"), "w"), indent=1)
