#!/usr/bin/env python3
"""Smoke test L0-b: carica il Go2 di mujoco_menagerie, tiene la posa "home" con un PD
in Python (stessa convenzione di Isaac Lab/unitree_rl_gym: tau = Kp*(q_des-q) - Kd*qd),
applica una spinta e misura la velocita' di simulazione su CPU.

Uso:
    python scripts/smoke_go2_mujoco.py --menagerie third_party/mujoco_menagerie [--seconds 3]

Criterio di successo (go/no-go per il setup MuJoCo):
  - il modello si carica (14 nq totali: 7 base + 12 giunti... nq=19, nv=18, nu=12);
  - dopo 2 s di PD la base resta sopra 0.20 m e |roll|,|pitch| < 0.2 rad;
  - dopo la spinta (0.5 m/s laterale) la base NON deve necessariamente restare in piedi
    (un PD statico non e' una policy): il test stampa solo cosa succede;
  - stampa gli step/s di physics (dt=0.005) per stimare il budget CPU.
"""
import argparse
import math
import os
import time

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover
    raise SystemExit("mujoco non installato: pip install mujoco") from e

# Posa di default (rad) nell'ordine dei giunti del MJCF: FL, FR, RL, RR x (hip, thigh, calf).
# E' la keyframe "home" di menagerie (0, 0.9, -1.8). Isaac Lab usa (±0.1, 0.8/1.0, -1.5):
# la differenza e' solo di comodo; il piano fissa UNA posa e la usa ovunque (sim, sim2sim, reale).
HOME_QPOS_JOINTS = np.array([0.0, 0.9, -1.8] * 4)
KP, KD = 25.0, 0.5  # Isaac Lab UNITREE_GO2_CFG: stiffness 25, damping 0.5


def quat_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menagerie", default="third_party/mujoco_menagerie")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--push-at", type=float, default=2.0, help="istante della spinta (s)")
    ap.add_argument("--push-vel", type=float, default=0.5, help="velocita' impressa alla base (m/s, asse y)")
    args = ap.parse_args()

    xml = os.path.join(args.menagerie, "unitree_go2", "scene.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.005  # come Isaac Lab (sim.dt=0.005, decimation 4 -> policy 50 Hz)
    print(f"model: nq={model.nq} nv={model.nv} nu={model.nu} dt={model.opt.timestep}")
    assert model.nu == 12 and model.nv == 18, "atteso Go2 con 12 attuatori e base libera"

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)

    n_steps = int(args.seconds / model.opt.timestep)
    push_step = int(args.push_at / model.opt.timestep)
    pushed = False
    t0 = time.perf_counter()
    for i in range(n_steps):
        q = data.qpos[7:]
        qd = data.qvel[6:]
        tau = KP * (HOME_QPOS_JOINTS - q) - KD * qd
        data.ctrl[:] = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        if i == push_step and args.push_vel != 0.0:
            data.qvel[1] += args.push_vel  # "push_by_setting_velocity" come in Isaac Lab
            pushed = True
        mujoco.mj_step(model, data)
        if i == int(2.0 / model.opt.timestep) - 1:
            r, p, _ = quat_to_rpy(data.qpos[3:7])
            h = data.qpos[2]
            print(f"t=2.0s  base_h={h:.3f} m  roll={r:+.3f} pitch={p:+.3f} rad")
            ok = h > 0.20 and abs(r) < 0.2 and abs(p) < 0.2
            print("PD hold:", "OK" if ok else "FALLITO (controllare Kp/Kd/posa)")
    dt_wall = time.perf_counter() - t0
    r, p, _ = quat_to_rpy(data.qpos[3:7])
    print(f"t={args.seconds:.1f}s base_h={data.qpos[2]:.3f} m roll={r:+.3f} pitch={p:+.3f} (spinta applicata: {pushed})")
    sps = n_steps / dt_wall
    print(f"physics steps/s (1 env, CPU): {sps:,.0f}  -> policy steps/s a decimation 4: {sps/4:,.0f}")


if __name__ == "__main__":
    main()
