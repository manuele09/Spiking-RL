"""Confronto CPU (MuJoCo) vs GPU (MJX) di Go2Balance con azione zero (solo PD).
Uso (sul PC GPU): python scripts/validate_mjx.py [--envs 512]
Riferimento CPU (CLAUDE.md, 30 ep, spinta fissa ogni 1.5-3 s): sopravvivenza a 10 s = 30/30 (0.5), 15/30 (1.0), 0/30 (1.5).
"""
import argparse
import time

import numpy as np
import torch

from spiking_rl.envs.mjx_go2 import Go2BalanceMJX


def zero_policy_survival(mag, n, seed=1, **kw):
    env = Go2BalanceMJX(n, seed=seed, push_vel=(mag, mag), push_every=(1.5, 3.0), **kw)
    env.reset()
    z = torch.zeros(n, 12, device="cuda")
    alive = torch.ones(n, dtype=torch.bool, device="cuda")
    surv = torch.zeros(n, dtype=torch.bool, device="cuda")
    for _ in range(env.max_steps):
        _, _, _, term, trunc, _, _ = env.step(z)
        surv |= alive & trunc & ~term
        alive &= ~(term | trunc)
    return float(surv.float().mean())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--cone", default="pyramidal")
    p.add_argument("--foot-condim", type=int, default=None)
    a = p.parse_args()
    kw = dict(cone=a.cone, foot_condim=a.foot_condim)
    print("config:", kw)
    # 1) quota a riposo (PD, nessuna spinta, nessuna randomizzazione)
    env = Go2BalanceMJX(4, push=False, randomize=False, **kw)
    env.reset()
    z = torch.zeros(4, 12, device="cuda")
    t0 = time.time()
    for _ in range(100):
        out = env.step(z)
    print("quota base dopo 2 s a riposo (MJX): %.4f m  | h_ref %.4f  (CPU: la policy compensa il cedimento di ~6 cm)"
          % (float(env.state.data.qpos[0, 2]), env.h_ref))
    for mag in (0.5, 1.0, 1.5):
        t0 = time.time()
        s = zero_policy_survival(mag, a.envs, **kw)
        print(f"azione zero, spinta {mag} m/s: sopravvivenza {s*100:.1f}% su {a.envs} env  ({time.time()-t0:.0f}s, incl. compilazione)")
