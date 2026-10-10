"""Test di robustezza a condizioni fisse (L5 ridotto, solo quelle supportate dall'env L1). Uso:
python -m spiking_rl.eval.robustness runs/go2_l1D_ann_s0 [--episodes 20] [--push 1.0]
Assi: attrito, carico (kg), latenza (passi da 20 ms), rumore osservazioni (x training), guasto Kp -30% su una gamba.
Tutte le condizioni con spinta fissa (default 1.0 m/s ogni 1.5-3 s); metrica: sopravvivenza su episodi da 10 s.
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

import spiking_rl.envs  # noqa: F401
from spiking_rl.eval.push_curve import load

AXES = {
    "friction": [0.3, 0.5, 0.8, 1.25],
    "payload": [2.0, 5.0, 8.0],
    "delay_steps": [0, 1, 2, 3],
    "obs_noise_scale": [1.0, 2.0, 4.0],
    "fault_leg": [0, 1, 2, 3],
}


def survival(agent, ck, kwargs, episodes, push):
    env = gym.make("Go2Balance-v0", push_vel=(push, push), push_every=(1.5, 3.0), **kwargs)
    s = 0
    for ep in range(episodes):
        o, _ = env.reset(seed=90_000 + ep)
        while True:
            x = torch.as_tensor(np.clip((o[None] - ck["norm_mean"]) / np.sqrt(ck["norm_var"] + 1e-8), -10, 10),
                                dtype=torch.float32)
            with torch.no_grad():
                a, _, _ = agent.act(x, deterministic=True)
            o, r, te, tr, _ = env.step(a[0].numpy())
            if te or tr:
                break
        s += int(tr and not te)
    return s / episodes


def run(run_dir, episodes, push):
    agent, ck, _ = load(run_dir)
    out = {"push": push, "episodes": episodes, "nominal": survival(agent, ck, {}, episodes, push)}
    print("nominale", out["nominal"])
    for axis, levels in AXES.items():
        out[axis] = {}
        for lv in levels:
            out[axis][str(lv)] = survival(agent, ck, {axis: lv}, episodes, push)
            print(f"{axis}={lv}: {out[axis][str(lv)]:.2f}", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--push", type=float, default=1.0)
    a = p.parse_args()
    json.dump(run(a.run_dir, a.episodes, a.push), open(os.path.join(a.run_dir, "robustness.json"), "w"), indent=1)
