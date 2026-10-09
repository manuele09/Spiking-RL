#!/usr/bin/env python3
"""Micro-benchmark per scegliere la libreria SNN (sezione 3.5 del piano).

Confronta, a parita' di rete (obs 45 -> 256 -> 256 -> 12, LIF, T interni, batch = num_env),
il tempo di forward+backward di:
  - "custom": LIF in PyTorch puro (scripts/smoke_lif_surrogate.py);
  - "snntorch": snntorch.Leaky (se installato);
  - "spikingjelly": spikingjelly.activation_based LIFNode in step_mode 'm' (se installato).
Le librerie assenti vengono saltate, non e' un errore.

Uso: python scripts/bench_snn_libs.py --batch 4096 --T 4 --iters 20
Da eseguire SULLA GPU TARGET prima di decidere: su CPU i numeri servono solo come sanity.
Criterio: scegliere la libreria entro 1.5x del migliore in tempo; a parita', preferire la
piu' semplice da esportare (custom > snntorch > spikingjelly) perche' il deploy su robot/NIR
richiede un grafo esplicito e senza kernel custom.
"""
import argparse
import importlib
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_lif_surrogate import SpikingActor  # noqa: E402


def bench(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def make_custom(n_obs, n_act, h, T, dev):
    net = SpikingActor(n_obs, n_act, (h, h), T=T).to(dev)

    def step(obs):
        out = net(obs)
        (out ** 2).mean().backward()
    return step


def make_snntorch(n_obs, n_act, h, T, dev):
    snn = importlib.import_module("snntorch")
    surrogate = importlib.import_module("snntorch.surrogate")
    sg = surrogate.atan()
    fc1, fc2, out = nn.Linear(n_obs, h), nn.Linear(h, h), nn.Linear(h, n_act)
    l1 = snn.Leaky(beta=0.5, spike_grad=sg)
    l2 = snn.Leaky(beta=0.5, spike_grad=sg)
    mods = nn.ModuleList([fc1, fc2, out, l1, l2]).to(dev)

    def step(obs):
        m1, m2 = l1.init_leaky(), l2.init_leaky()
        acc = 0.0
        for _ in range(T):
            s1, m1 = l1(fc1(obs), m1)
            s2, m2 = l2(fc2(s1), m2)
            acc = acc + out(s2)
        ((acc / T) ** 2).mean().backward()
    return step


def make_spikingjelly(n_obs, n_act, h, T, dev):
    neuron = importlib.import_module("spikingjelly.activation_based.neuron")
    functional = importlib.import_module("spikingjelly.activation_based.functional")
    layer = importlib.import_module("spikingjelly.activation_based.layer")
    net = nn.Sequential(
        layer.Linear(n_obs, h), neuron.LIFNode(tau=2.0, step_mode="m"),
        layer.Linear(h, h), neuron.LIFNode(tau=2.0, step_mode="m"),
        layer.Linear(h, n_act),
    ).to(dev)
    functional.set_step_mode(net, "m")

    def step(obs):
        functional.reset_net(net)
        x = obs.unsqueeze(0).repeat(T, 1, 1)  # [T, B, obs]
        out = net(x).mean(0)
        (out ** 2).mean().backward()
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    obs = torch.randn(args.batch, 45, device=dev)
    print(f"device={dev} batch={args.batch} T={args.T} hidden={args.hidden} iters={args.iters}")
    for name, maker in [("custom", make_custom), ("snntorch", make_snntorch), ("spikingjelly", make_spikingjelly)]:
        try:
            step = maker(45, 12, args.hidden, args.T, dev)
        except ImportError:
            print(f"{name:13s}: non installata (saltata)")
            continue
        except Exception as e:  # API cambiata: segnalare, non bloccare
            print(f"{name:13s}: errore di costruzione -> {type(e).__name__}: {e}")
            continue
        ms = bench(lambda: step(obs), args.iters)
        print(f"{name:13s}: {ms:8.2f} ms / fwd+bwd")


if __name__ == "__main__":
    main()
