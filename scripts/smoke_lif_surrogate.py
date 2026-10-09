#!/usr/bin/env python3
"""Smoke test L0-a: strato LIF minimale in PyTorch puro con surrogate gradient,
multi-step (T passi interni per ogni passo di controllo) e decodifica da potenziale di membrana.

Verifica (go/no-go per il "core SNN" del piano):
  1. forward: firing rate per strato in (0.05, 0.6) con input normalizzati N(0,1);
  2. backward: il gradiente arriva al primo strato (norma > 0) e non esplode (< 1e3);
  3. la pendenza del surrogate e' un parametro modificabile a runtime (per lo scheduling);
  4. lo stato di membrana puo' essere portato tra passi di controllo (carry=True) e resettato
     per gli env terminati tramite maschera dones (come fa rsl_rl per GRU/LSTM).

Uso: python scripts/smoke_lif_surrogate.py [--T 4] [--hidden 256] [--batch 4096]
Costo: < 10 s su CPU.
"""
import argparse
import time

import torch
import torch.nn as nn


class SpikeFn(torch.autograd.Function):
    """Heaviside con surrogate 'arctan' (SpikingJelly default): dS/du ~ alpha/2 / (1 + (pi/2 alpha u)^2)."""

    @staticmethod
    def forward(ctx, u, alpha):
        ctx.save_for_backward(u)
        ctx.alpha = alpha
        return (u >= 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        a = ctx.alpha
        sg = (a / 2) / (1 + (torch.pi / 2 * a * u) ** 2)
        return g * sg, None


class LIFLayer(nn.Module):
    """LIF a corrente istantanea, leak appreso per strato (sigmoide), reset hard, soglia 1."""

    def __init__(self, n_in, n_out, tau_init=2.0, in_second_moment=1.0):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        # Inizializzazione "rate-aware": con l'init di default di nn.Linear il secondo strato
        # (ingresso = spike con rate ~0.1) non raggiunge mai la soglia -> strato morto.
        # Si scala std(W) in modo che la corrente abbia varianza ~1 dato E[x^2] dell'ingresso
        # (1.0 per osservazioni normalizzate, ~rate per ingressi spiking).
        nn.init.normal_(self.fc.weight, std=1.0 / (n_in * in_second_moment) ** 0.5)
        nn.init.zeros_(self.fc.bias)
        # beta = sigmoid(w) in (0,1): decadimento della membrana; init da tau
        self.w_beta = nn.Parameter(torch.full((n_out,), float(torch.logit(torch.tensor(1 - 1 / tau_init)))))
        self.alpha = 2.0  # pendenza surrogate (modificabile da fuori: scheduling)
        self.register_buffer("u", torch.zeros(0))  # stato di membrana (batch, n_out)

    def reset(self, batch, device, dones=None):
        if self.u.numel() == 0 or self.u.shape[0] != batch:
            self.u = torch.zeros(batch, self.fc.out_features, device=device)
        elif dones is not None:
            self.u = self.u * (1 - dones.view(-1, 1).to(self.u.dtype))
        else:
            self.u = torch.zeros_like(self.u)

    def forward(self, x):
        beta = torch.sigmoid(self.w_beta)
        u = beta * self.u + self.fc(x)
        s = SpikeFn.apply(u - 1.0, self.alpha)
        self.u = u * (1 - s)  # reset hard
        return s


class SpikingActor(nn.Module):
    """obs -> [LIF x L] -> neurone di uscita non-spiking (membrana integrata, media su T)."""

    def __init__(self, n_obs, n_act, hidden=(256, 256), T=4, carry=False):
        super().__init__()
        dims = (n_obs,) + tuple(hidden)
        self.layers = nn.ModuleList(
            LIFLayer(dims[i], dims[i + 1], in_second_moment=(1.0 if i == 0 else 0.15))
            for i in range(len(hidden))
        )
        self.out = nn.Linear(dims[-1], n_act)
        self.T, self.carry = T, carry
        self.last_rates = []

    def reset(self, batch, device, dones=None):
        for l in self.layers:
            l.reset(batch, device, dones)

    def set_alpha(self, a):
        for l in self.layers:
            l.alpha = a

    def forward(self, obs):
        if not self.carry:
            self.reset(obs.shape[0], obs.device)
        acc = 0.0
        rates = [0.0] * len(self.layers)
        for _ in range(self.T):
            x = obs  # encoding diretto: la stessa osservazione e' iniettata a ogni passo interno
            for i, l in enumerate(self.layers):
                x = l(x)
                rates[i] += x.mean().item() / self.T
            acc = acc + self.out(x)
        self.last_rates = rates
        return acc / self.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--n-obs", type=int, default=45)
    ap.add_argument("--n-act", type=int, default=12)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = SpikingActor(args.n_obs, args.n_act, (args.hidden, args.hidden), T=args.T).to(dev)
    obs = torch.randn(args.batch, args.n_obs, device=dev)

    t0 = time.perf_counter()
    out = net(obs)
    loss = (out ** 2).mean()
    loss.backward()
    dt = time.perf_counter() - t0
    g0 = net.layers[0].fc.weight.grad.norm().item()
    gmax = max(p.grad.norm().item() for p in net.parameters() if p.grad is not None)
    print(f"device={dev} T={args.T} batch={args.batch} hidden={args.hidden}  fwd+bwd={dt*1e3:.1f} ms")
    print(f"firing rate per strato: {[round(r, 3) for r in net.last_rates]}")
    print(f"grad norm primo strato={g0:.3e}  max grad norm={gmax:.3e}")
    ok_rates = all(0.05 < r < 0.6 for r in net.last_rates)
    ok_grad = g0 > 0 and gmax < 1e3
    # scheduling della pendenza + carry di stato con reset per maschera dones
    net.set_alpha(0.5)
    net.carry = True
    net.reset(args.batch, dev)
    _ = net(obs)
    dones = (torch.rand(args.batch, device=dev) < 0.1).float()
    net.reset(args.batch, dev, dones=dones)
    u_done = net.layers[0].u[dones.bool()].abs().sum().item()
    ok_carry = u_done == 0.0
    print(f"carry+reset per dones: {'OK' if ok_carry else 'FALLITO'}  (alpha ora {net.layers[0].alpha})")
    print("RISULTATO:", "OK" if (ok_rates and ok_grad and ok_carry) else "FALLITO",
          "(rates ok:", ok_rates, "grad ok:", ok_grad, ")")


if __name__ == "__main__":
    main()
