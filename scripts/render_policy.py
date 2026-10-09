#!/usr/bin/env python3
"""Video di un episodio L1 (spinte a intensita' fissa) per una policy salvata o per l'azione zero.
Uso: python scripts/render_policy.py --out docs/video/x.mp4 [--run runs/go2_l1_annC_s0] [--push 1.0] [--seed 3]
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
import gymnasium as gym
import imageio.v2 as iio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

import spiking_rl.envs  # noqa: F401
from spiking_rl.eval.push_curve import load


def episode_frames(run, push, seed, label, w=640, h=480, fps=25):
    env = gym.make("Go2Balance-v0", push_vel=(push, push), push_every=(1.5, 3.0))
    u = env.unwrapped
    agent, ck, _ = load(run) if run else (None, None, None)
    r = mujoco.Renderer(u.m, h, w)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 140, -15, 2.2
    o, _ = env.reset(seed=seed)
    frames, step, fell, pushed_flash = [], 0, False, 0
    every = max(1, int(round(1 / (u.dt * fps))))
    while True:
        if agent is None:
            a = np.zeros(12)
        else:
            x = torch.as_tensor(np.clip((o[None] - ck["norm_mean"]) / np.sqrt(ck["norm_var"] + 1e-8), -10, 10), dtype=torch.float32)
            with torch.no_grad():
                a = agent.act(x, deterministic=True)[0][0].numpy()
        o, rew, te, tr, info = env.step(a)
        pushed_flash = 8 if info["pushed"] else max(0, pushed_flash - 1)
        if step % every == 0:
            cam.lookat[:] = [u.d.qpos[0], u.d.qpos[1], 0.2]
            r.update_scene(u.d, cam)
            im = Image.fromarray(r.render())
            dr = ImageDraw.Draw(im)
            dr.text((10, 10), f"{label}   t={step * u.dt:4.1f}s   push {push:.1f} m/s", fill=(255, 255, 255))
            if pushed_flash:
                dr.text((10, 28), "SPINTA!", fill=(255, 80, 80))
            if te:
                dr.text((10, 46), "CADUTO", fill=(255, 80, 80))
            frames.append(np.asarray(im))
        step += 1
        if te or tr:
            frames += [frames[-1]] * fps  # 1 s di pausa sull'ultimo frame
            break
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--push", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--compare-zero", action="store_true", help="prima l'azione zero, poi la policy")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    frames = []
    if a.compare_zero or a.run is None:
        frames += episode_frames(None, a.push, a.seed, "Azione zero (solo PD)")
    if a.run:
        frames += episode_frames(a.run, a.push, a.seed, f"Policy {os.path.basename(a.run)}")
    iio.mimsave(a.out, frames, fps=25, macro_block_size=1)
    print("scritto", a.out, len(frames), "frame")
