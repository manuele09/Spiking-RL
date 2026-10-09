#!/usr/bin/env python3
"""Render offscreen del Go2 (MuJoCo) -> PNG in docs/img. Richiede libosmesa6 (MUJOCO_GL=osmesa)."""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
import imageio.v2 as iio
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("third_party/mujoco_menagerie/unitree_go2/scene.xml")
d = mujoco.MjData(m)
m.opt.timestep = 0.005
Q = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])  # posa Isaac Lab
d.qpos[2], d.qpos[3], d.qpos[7:] = 0.32, 1, Q
r = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
cam.azimuth, cam.elevation, cam.distance = 140, -15, 1.8
cam.lookat[:] = [0, 0, 0.25]
os.makedirs("docs/img", exist_ok=True)


def shot(name):
    mujoco.mj_forward(m, d)
    cam.lookat[:2] = d.qpos[:2]
    r.update_scene(d, cam)
    iio.imwrite(f"docs/img/{name}.png", r.render())


def run(n):
    for _ in range(n):
        d.ctrl[:] = np.clip(25 * (Q - d.qpos[7:]) - 0.5 * d.qvel[6:], -23.5, 23.5)
        mujoco.mj_step(m, d)


shot("go2_01_inizio")
run(400)
shot("go2_02_pd_posa_default")
d.qvel[1] += 1.0
run(60)
shot("go2_03_dopo_spinta_1ms")
run(400)
shot("go2_04_dopo_recupero")
print("ok", d.qpos[:3])
