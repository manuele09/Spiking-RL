"""Conteggio operazioni ed energia stimata per inferenza (Appendice B del piano). Uso:
python -m spiking_rl.eval.synops runs/go2_l1D_snn_T4_s0 [--episodes 5]

Convenzione 45 nm (Horowitz 2014): 0.9 pJ per AC (accumulo di uno spike), 4.6 pJ per MAC.
Non sono misure su silicio: ignorano memoria, traffico, overhead e aggiornamento dei neuroni.
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

import spiking_rl.envs  # noqa: F401
from spiking_rl.eval.push_curve import load

E_AC, E_MAC = 0.9, 4.6  # pJ


def ann_ops(actor):
    macs = sum(m.in_features * m.out_features for m in actor.modules() if isinstance(m, torch.nn.Linear))
    return {"mac": macs, "ac": 0, "energy_pj": macs * E_MAC}


def snn_ops(actor, spike_counts):
    """Full accounting. Strato 1: ingresso analogico -> MAC (contati T volte, conservativo: la corrente
    W*obs e' costante nei T passi e potrebbe essere calcolata una volta sola). Strati successivi e readout:
    ingressi binari -> AC = spike_in x fan_out. L'encoder diretto non aggiunge operazioni."""
    T = actor.T
    l1 = actor.layers[0].fc
    mac_conservative = T * l1.in_features * l1.out_features
    mac_once = l1.in_features * l1.out_features
    ac = 0.0
    for i in range(1, len(actor.layers)):
        ac += spike_counts[i - 1] * actor.layers[i].fc.out_features
    ac += spike_counts[-1] * actor.out.out_features
    return {
        "mac_conservative": mac_conservative, "mac_once": mac_once, "ac": ac,
        "energy_pj_conservative": mac_conservative * E_MAC + ac * E_AC,
        "energy_pj_optimized": mac_once * E_MAC + ac * E_AC,
        "synops_only_ac": ac,
    }


def run(run_dir, episodes=5):
    agent, ck, c = load(run_dir)
    actor = agent.actor
    out = {"run": os.path.basename(run_dir), "actor": c["actor"]}
    if not actor.spiking:
        out.update(ann_ops(actor))
    else:
        actor.record_spikes = True
        env = gym.make("Go2Balance-v0", push_vel=(1.0, 1.0), push_every=(1.5, 3.0))
        sums, n = None, 0
        for ep in range(episodes):
            o, _ = env.reset(seed=70_000 + ep)
            while True:
                x = torch.as_tensor(np.clip((o[None] - ck["norm_mean"]) / np.sqrt(ck["norm_var"] + 1e-8), -10, 10),
                                    dtype=torch.float32)
                with torch.no_grad():
                    a, _, _ = agent.act(x, deterministic=True)
                sc = np.array(actor.spike_counts)
                sums = sc if sums is None else sums + sc
                n += 1
                o, r, te, tr, _ = env.step(a[0].numpy())
                if te or tr:
                    break
        mean_counts = (sums / n).tolist()
        out["mean_spikes_per_inference_per_layer"] = mean_counts
        out["T"] = actor.T
        out.update(snn_ops(actor, mean_counts))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--episodes", type=int, default=5)
    a = p.parse_args()
    res = run(a.run_dir, a.episodes)
    json.dump(res, open(os.path.join(a.run_dir, "synops.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
