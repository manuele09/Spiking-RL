"""Go2Balance su GPU con MJX (JAX): stesse convenzioni, osservazioni, reward e randomizzazioni di
`mujoco_go2/env.py`, ma batched (num_envs sulla GPU) e con autoreset in stile Gymnasium SAME_STEP.

Stesso modello MJCF della versione CPU (menagerie `scene.xml`): cambiano solo il solver (iterazioni ridotte per MJX) e
la libreria di collisioni. Le due versioni NON sono identiche numericamente: va validata la corrispondenza
(`scripts/validate_mjx.py`).

API: step/reset ritornano tensori torch sulla stessa GPU (via DLPack, senza copie in memoria host).
"""
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import torch
from mujoco import mjx

MENAGERIE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "mujoco_menagerie")
Q_DEFAULT = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
KP, KD, TAU_MAX = 25.0, 0.5, 23.5
ACT_SCALE, ACT_CLIP = 0.25, 4.0
DT, DECIMATION = 0.005, 4
D_MAX = 4  # ritardo massimo supportato (passi di policy)
OBS_NOISE = np.concatenate([np.full(3, 0.2), np.full(3, 0.05), np.zeros(3), np.full(12, 0.01),
                            np.full(12, 0.15), np.zeros(12)]).astype(np.float32)


class EnvState(NamedTuple):
    data: mjx.Data
    last_a: jax.Array   # azione (clippata) del passo precedente: parte dell'osservazione e termine di action-rate
    hist: jax.Array     # (D_MAX, 12) azioni passate, hist[0] = piu' recente
    last_qd: jax.Array
    delay: jax.Array
    kp: jax.Array
    kd: jax.Array
    mass: jax.Array
    fric: jax.Array
    step_i: jax.Array
    next_push: jax.Array
    key: jax.Array
    ep_ret: jax.Array


def _rot_inv(q, v):
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ v


def _to_torch(x):
    return torch.from_dlpack(x)


class Go2BalanceMJX:
    obs_dim, act_dim = 45, 12

    def __init__(self, num_envs, seed=0, push=True, randomize=True, episode_s=10.0, push_vel=(0.3, 1.0),
                 push_every=(3.0, 6.0), w_pose=0.5, w_action_rate=0.01, term_penalty=0.0, friction=None,
                 payload=None, delay_steps=None, obs_noise_scale=1.0, fault_leg=None, fault_kp_scale=0.7,
                 iterations=8, ls_iterations=8, cone="pyramidal", foot_condim=None):
        m = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE, "unitree_go2", "scene.xml"))
        m.opt.timestep = DT
        m.opt.iterations, m.opt.ls_iterations = iterations, ls_iterations
        # il cono ellittico della versione CPU e' ~3.5x piu' lento in MJX: default piramidale (validato in validate_mjx.py)
        m.opt.cone = {"elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC, "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL}[cone]
        if foot_condim is not None:
            for n in ("FL", "FR", "RL", "RR"):
                m.geom_condim[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)] = foot_condim
        # solo collisioni robot-suolo: le auto-collisioni (es. cilindro-box) non sono supportate da MJX e sono irrilevanti
        # per questi compiti. Differenza voluta rispetto alla versione CPU.
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        coll = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        m.geom_contype[coll], m.geom_conaffinity[coll] = 1, 0
        m.geom_contype[floor], m.geom_conaffinity[floor] = 0, 1
        self.m = m
        self.mx = mjx.put_model(m)
        self.N = num_envs
        self.push, self.randomize = push, randomize
        self.push_vel, self.push_every = push_vel, push_every
        self.w_pose, self.w_action_rate, self.term_penalty = w_pose, w_action_rate, term_penalty
        self.fix_friction, self.fix_payload, self.fix_delay = friction, payload, delay_steps
        self.obs_noise_scale, self.fault_leg, self.fault_kp_scale = obs_noise_scale, fault_leg, fault_kp_scale
        self.max_steps = int(episode_s / (DT * DECIMATION))
        self.dt = DT * DECIMATION
        self.base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("FL", "FR", "RL", "RR")]
        self.is_foot = np.zeros(m.ngeom, bool)
        self.is_foot[feet] = True
        self.mass0 = float(m.body_mass[self.base_id])
        self.fric0 = float(m.geom_friction[self.floor_id, 0])
        d = mujoco.MjData(m)
        d.qpos[2], d.qpos[3], d.qpos[7:] = 0.5, 1, Q_DEFAULT
        mujoco.mj_forward(m, d)
        self.h_ref = float(0.5 - min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in feet))
        self.d0 = mjx.make_data(m)
        self.key = jax.random.PRNGKey(seed)
        self._reset_v = jax.jit(jax.vmap(self._reset_one))
        self._step_v = jax.jit(jax.vmap(self._step_one))
        self.state = None

    # --- osservazione ---
    def _obs(self, d, last_a, key):
        g = _rot_inv(d.qpos[3:7], jnp.array([0.0, 0.0, -1.0]))
        o = jnp.concatenate([d.qvel[3:6], g, jnp.zeros(3), d.qpos[7:] - Q_DEFAULT, d.qvel[6:], last_a])
        if self.randomize:
            o = o + jax.random.uniform(key, (45,), minval=-1.0, maxval=1.0) * self.obs_noise_scale * OBS_NOISE
        return o.astype(jnp.float32)

    def _next_push(self, key, t):
        return t + jax.random.uniform(key, (), minval=self.push_every[0], maxval=self.push_every[1])

    # --- reset di un singolo env ---
    def _reset_one(self, key):
        k = jax.random.split(key, 12)
        z = jnp.zeros(12)
        q0 = jnp.asarray(Q_DEFAULT)
        tilt = jnp.zeros(2)
        mass, fric, kp, kd, delay = self.mass0, self.fric0, jnp.full(12, KP), jnp.full(12, KD), jnp.int32(0)
        if self.randomize:
            q0 = q0 + jax.random.uniform(k[0], (12,), minval=-0.1, maxval=0.1)
            tilt = jnp.deg2rad(5.0) * jax.random.uniform(k[1], (2,), minval=-1.0, maxval=1.0)
            mass = self.mass0 + jax.random.uniform(k[2], (), minval=-1.0, maxval=3.0)
            fric = jax.random.uniform(k[3], (), minval=0.5, maxval=1.25)
            kp = KP * jax.random.uniform(k[4], (12,), minval=0.9, maxval=1.1)
            kd = KD * jax.random.uniform(k[5], (12,), minval=0.9, maxval=1.1)
            delay = jax.random.randint(k[6], (), 0, 2)
        if self.fix_payload is not None:
            mass = self.mass0 + self.fix_payload
        if self.fix_friction is not None:
            fric = jnp.float32(self.fix_friction)
        if self.fix_delay is not None:
            delay = jnp.int32(self.fix_delay)
        if self.fault_leg is not None:
            kp = kp.at[3 * self.fault_leg:3 * self.fault_leg + 3].multiply(self.fault_kp_scale)
        cr, sr, cp, sp = jnp.cos(tilt[0] / 2), jnp.sin(tilt[0] / 2), jnp.cos(tilt[1] / 2), jnp.sin(tilt[1] / 2)
        quat = jnp.array([cr * cp, sr * cp, cr * sp, -sr * sp])
        qpos = self.d0.qpos.at[2].set(self.h_ref + 0.02).at[3:7].set(quat).at[7:].set(q0)
        d = self.d0.replace(qpos=qpos, qvel=jnp.zeros_like(self.d0.qvel), time=jnp.zeros(()),
                            ctrl=jnp.zeros_like(self.d0.ctrl))
        last_a = z
        return EnvState(d, last_a, jnp.zeros((D_MAX, 12)), jnp.zeros(12), delay, kp, kd, mass, fric,
                        jnp.int32(0), self._next_push(k[7], 0.0), k[8], jnp.zeros(())), \
            self._obs(d, last_a, k[9])

    # --- passo di un singolo env ---
    def _step_one(self, s: EnvState, action):
        k = jax.random.split(s.key, 8)
        a = jnp.clip(action, -ACT_CLIP, ACT_CLIP)
        applied = jnp.where(s.delay == 0, a, s.hist[jnp.maximum(s.delay - 1, 0)])
        hist = jnp.concatenate([a[None], s.hist[:-1]], 0)
        q_des = jnp.asarray(Q_DEFAULT) + ACT_SCALE * applied
        mx = self.mx.replace(body_mass=self.mx.body_mass.at[self.base_id].set(s.mass),
                             geom_friction=self.mx.geom_friction.at[self.floor_id, 0].set(s.fric))

        def sub(d, _):
            tau = jnp.clip(s.kp * (q_des - d.qpos[7:]) - s.kd * d.qvel[6:], -TAU_MAX, TAU_MAX)
            d = mjx.step(mx, d.replace(ctrl=tau))
            return d, jnp.sum(tau ** 2)

        d, taus = jax.lax.scan(sub, s.data, None, length=DECIMATION)
        tau_sum = jnp.sum(taus)
        step_i = s.step_i + 1
        t = step_i * self.dt
        next_push = s.next_push
        if self.push:
            do = t >= s.next_push
            mag = jax.random.uniform(k[0], (), minval=self.push_vel[0], maxval=self.push_vel[1])
            ang = jax.random.uniform(k[1], (), minval=0.0, maxval=2 * jnp.pi)
            dv = jnp.where(do, 1.0, 0.0) * mag * jnp.array([jnp.cos(ang), jnp.sin(ang)])
            d = d.replace(qvel=d.qvel.at[:2].add(dv))
            next_push = jnp.where(do, self._next_push(k[2], t), s.next_push)
        qd = d.qvel[6:]
        g = _rot_inv(d.qpos[3:7], jnp.array([0.0, 0.0, -1.0]))
        v_b = _rot_inv(d.qpos[3:7], d.qvel[:3])
        h = d.qpos[2]
        c = d._impl.contact
        floor_pair = (c.geom1 == self.floor_id) | (c.geom2 == self.floor_id)
        other = jnp.where(c.geom1 == self.floor_id, c.geom2, c.geom1)
        foot = jnp.asarray(self.is_foot)[jnp.clip(other, 0, self.m.ngeom - 1)]
        bad = jnp.sum(floor_pair & ~foot & (c.dist < c.includemargin) & (c.geom1 >= 0))
        r = (-2.5 * jnp.sum(g[:2] ** 2) - 20.0 * (h - self.h_ref) ** 2 - 1.0 * jnp.sum(v_b[:2] ** 2)
             - 2.0 * v_b[2] ** 2 - 0.05 * jnp.sum(d.qvel[3:6] ** 2) - self.w_pose * jnp.sum((d.qpos[7:] - Q_DEFAULT) ** 2)
             - 2e-4 * tau_sum / DECIMATION - self.w_action_rate * jnp.sum((a - s.last_a) ** 2)
             - 2.5e-7 * jnp.sum(((qd - s.last_qd) / self.dt) ** 2) - 1.0 * jnp.minimum(bad, 1) + 0.5) * self.dt
        tilt = jnp.arccos(jnp.clip(-g[2], -1.0, 1.0))
        # stato non finito (instabilita' del solver con pochi iterazioni): termina e resetta invece di propagare i NaN
        nonfinite = ~(jnp.all(jnp.isfinite(d.qpos)) & jnp.all(jnp.isfinite(d.qvel)))
        term = (bad > 0) | (tilt > 1.0) | (h < 0.15) | nonfinite
        r = jnp.nan_to_num(r, nan=-1.0)
        trunc = step_i >= self.max_steps
        r = r - jnp.where(term, self.term_penalty, 0.0)
        ep_ret = s.ep_ret + r
        s2 = EnvState(d, a, hist, qd, s.delay, s.kp, s.kd, s.mass, s.fric, step_i, next_push, k[3], ep_ret)
        final_obs = jnp.nan_to_num(self._obs(d, a, k[4]))
        done = term | trunc
        s_reset, obs_reset = self._reset_one(k[5])
        # autoreset: sostituisco solo lo stato di integrazione (il resto di Data viene ricalcolato da mjx.step)
        keep = ("qpos", "qvel", "time", "qacc_warmstart", "ctrl")
        dsel = d.replace(**{f: jnp.where(done, getattr(s_reset.data, f), getattr(d, f)) for f in keep})
        s_out = jax.tree_util.tree_map(lambda x, y: jnp.where(done, y, x), s2._replace(data=None),
                                       s_reset._replace(data=None))._replace(data=dsel)
        obs = jnp.where(done, obs_reset, final_obs)
        return s_out, obs, final_obs, r, term, trunc, ep_ret, tilt

    # --- API torch ---
    def reset(self):
        self.key, k = jax.random.split(self.key)
        self.state, obs = self._reset_v(jax.random.split(k, self.N))
        return _to_torch(obs)

    def step(self, action):
        """action: tensor torch (N,12) su GPU. Ritorna obs (post-reset), final_obs, r, term, trunc, ep_ret (alla fine
        dell'episodio, altrimenti ritorno parziale), tilt."""
        a = jnp.from_dlpack(action.detach().contiguous())
        self.state, obs, fobs, r, term, trunc, ep_ret, tilt = self._step_v(self.state, a)
        return tuple(_to_torch(x) for x in (obs, fobs, r, term, trunc, ep_ret, tilt))
