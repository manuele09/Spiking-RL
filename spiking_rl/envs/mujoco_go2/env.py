"""Go2 in MuJoCo (CPU) per L1: stand & balance con spinte. Convenzioni: docs/implementation_plan.md §1.0, §1.3.

Env Gymnasium singolo (obs 45-d, azione 12-d); la vettorizzazione si fa con gym.vector.AsyncVectorEnv.
Ordine dei giunti = MJCF menagerie (FL, FR, RL, RR) x (hip, thigh, calf). Nota: Isaac Lab usa un ordine diverso
(per tipo di giunto); serve una permutazione solo al deploy/sim2sim.
"""
import os

import gymnasium as gym
import mujoco
import numpy as np

MENAGERIE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "mujoco_menagerie")
Q_DEFAULT = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])  # posa Isaac Lab
KP, KD, TAU_MAX = 25.0, 0.5, 23.5
ACT_SCALE, ACT_CLIP = 0.25, 4.0
DT, DECIMATION = 0.005, 4


def _rot_inv(q, v):
    """Ruota v dal frame mondo al frame corpo (q = quaternione wxyz del corpo)."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ v


class Go2BalanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, push=True, randomize=True, episode_s=10.0, push_vel=(0.3, 1.0), push_every=(3.0, 6.0),
                 command_height=False, seed=None):
        self.m = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE, "unitree_go2", "scene.xml"))
        self.m.opt.timestep = DT
        self.d = mujoco.MjData(self.m)
        self.push, self.randomize = push, randomize
        self.push_vel, self.push_every = push_vel, push_every
        self.max_steps = int(episode_s / (DT * DECIMATION))
        self.base_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.floor_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geoms = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("FL", "FR", "RL", "RR")}
        self.mass0 = self.m.body_mass[self.base_id]
        self.fric0 = self.m.geom_friction[self.floor_id].copy()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (45,), np.float32)
        self.action_space = gym.spaces.Box(-ACT_CLIP, ACT_CLIP, (12,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.h_ref = self._default_height()
        self.dt = DT * DECIMATION

    def _default_height(self):
        d = mujoco.MjData(self.m)
        d.qpos[2], d.qpos[3], d.qpos[7:] = 0.5, 1, Q_DEFAULT
        mujoco.mj_forward(self.m, d)
        feet = [g for g in self.foot_geoms]
        low = min(d.geom_xpos[g][2] - self.m.geom_size[g][0] for g in feet)
        return float(0.5 - low)  # quota base a cui i piedi toccano il suolo senza flessione del PD

    # --- osservazione ---
    def _obs(self):
        d = self.d
        q = d.qpos[7:]
        qd = d.qvel[6:]
        w_body = d.qvel[3:6]  # in MuJoCo la velocita' angolare libera e' gia' nel frame locale
        g = _rot_inv(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        noise = self.rng.uniform(-1, 1, 45) if self.randomize else np.zeros(45)
        o = np.concatenate([w_body, g, np.zeros(3), q - Q_DEFAULT, qd, self.last_a])
        o = o + noise * np.concatenate([np.full(3, 0.2), np.full(3, 0.05), np.zeros(3),
                                        np.full(12, 0.01), np.full(12, 1.5 * 0.1), np.zeros(12)])
        return o.astype(np.float32)

    # --- Gymnasium ---
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        d, m = self.d, self.m
        mujoco.mj_resetData(m, d)
        q0 = Q_DEFAULT + (self.rng.uniform(-0.1, 0.1, 12) if self.randomize else 0)
        d.qpos[2], d.qpos[7:] = self.h_ref + 0.02, q0
        tilt = np.deg2rad(5.0) * self.rng.uniform(-1, 1, 2) if self.randomize else np.zeros(2)
        cr, sr, cp, sp = np.cos(tilt[0] / 2), np.sin(tilt[0] / 2), np.cos(tilt[1] / 2), np.sin(tilt[1] / 2)
        d.qpos[3:7] = [cr * cp, sr * cp, cr * sp, -sr * sp]
        self.kp = np.full(12, KP)
        self.kd = np.full(12, KD)
        m.body_mass[self.base_id] = self.mass0
        m.geom_friction[self.floor_id] = self.fric0
        self.delay = 0
        if self.randomize:
            m.body_mass[self.base_id] = self.mass0 + self.rng.uniform(-1, 3)
            m.geom_friction[self.floor_id, 0] = self.rng.uniform(0.5, 1.25)
            self.kp = KP * self.rng.uniform(0.9, 1.1, 12)
            self.kd = KD * self.rng.uniform(0.9, 1.1, 12)
            self.delay = int(self.rng.integers(0, 2))
        self.last_a = np.zeros(12)
        self.prev_a = np.zeros(12)
        self.a_buf = np.zeros(12)
        self.step_i = 0
        self.next_push = self._next_push_time()
        mujoco.mj_forward(m, d)
        self.last_qd = d.qvel[6:].copy()
        return self._obs(), {}

    def _next_push_time(self):
        return self.step_i * self.dt + self.rng.uniform(*self.push_every)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -ACT_CLIP, ACT_CLIP)
        applied = self.a_buf if self.delay else a  # ritardo di un passo di policy (20 ms)
        self.a_buf = a
        q_des = Q_DEFAULT + ACT_SCALE * applied
        d, m = self.d, self.m
        tau_sum = 0.0
        for _ in range(DECIMATION):
            tau = np.clip(self.kp * (q_des - d.qpos[7:]) - self.kd * d.qvel[6:], -TAU_MAX, TAU_MAX)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            tau_sum += float(np.sum(tau ** 2))
        self.step_i += 1
        pushed = False
        if self.push and self.step_i * self.dt >= self.next_push:
            mag = self.rng.uniform(*self.push_vel)
            ang = self.rng.uniform(0, 2 * np.pi)
            d.qvel[0] += mag * np.cos(ang)
            d.qvel[1] += mag * np.sin(ang)
            self.next_push = self._next_push_time()
            pushed = True
        qd = d.qvel[6:]
        g = _rot_inv(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        v_b = _rot_inv(d.qpos[3:7], d.qvel[:3])
        h = d.qpos[2]
        # contatti non-piede (base o coscia/anca col suolo)
        bad = 0
        for i in range(d.ncon):
            c = d.contact[i]
            pair = (c.geom1, c.geom2)
            if self.floor_id in pair:
                other = pair[0] if pair[1] == self.floor_id else pair[1]
                if other not in self.foot_geoms:
                    bad += 1
        # reward L1 (§1.3), tutti i termini x dt
        r = (-2.5 * np.sum(g[:2] ** 2) - 20.0 * (h - self.h_ref) ** 2 - 1.0 * np.sum(v_b[:2] ** 2)
             - 2.0 * v_b[2] ** 2 - 0.05 * np.sum(d.qvel[3:6] ** 2) - 0.5 * np.sum((d.qpos[7:] - Q_DEFAULT) ** 2)
             - 2e-4 * tau_sum / DECIMATION - 0.01 * np.sum((a - self.prev_a) ** 2)
             - 2.5e-7 * np.sum(((qd - self.last_qd) / self.dt) ** 2) - 1.0 * min(bad, 1) + 0.5) * self.dt
        self.prev_a, self.last_a, self.last_qd = a.copy(), a.copy(), qd.copy()
        roll_pitch_bad = np.arccos(np.clip(g[2] * -1.0, -1, 1)) > 1.0  # inclinazione totale > 1 rad
        terminated = bool(bad > 0 or roll_pitch_bad or h < 0.15)
        truncated = self.step_i >= self.max_steps
        info = {"pushed": pushed, "height": float(h), "tilt": float(np.arccos(np.clip(-g[2], -1, 1)))}
        return self._obs(), float(r), terminated, truncated, info
