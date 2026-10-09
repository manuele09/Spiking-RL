"""Neuroni spiking (PyTorch puro): LIF e ALIF con surrogate gradient.

Specifica: docs/implementation_plan.md, Appendice A.
"""
import torch
import torch.nn as nn


class SpikeFn(torch.autograd.Function):
    """Heaviside con surrogate arctan: dS/du ~ (a/2) / (1 + (pi/2 * a * u)^2)."""

    @staticmethod
    def forward(ctx, u, alpha):
        ctx.save_for_backward(u)
        ctx.alpha = alpha
        return (u >= 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        a = ctx.alpha
        return g * (a / 2) / (1 + (torch.pi / 2 * a * u) ** 2), None


def rate_aware_init_(linear: nn.Linear, in_second_moment: float = 1.0):
    """std(W) = 1/sqrt(fan_in * E[x^2]): senza questo il 2o strato resta morto (osservato)."""
    nn.init.normal_(linear.weight, std=1.0 / (linear.in_features * in_second_moment) ** 0.5)
    nn.init.zeros_(linear.bias)


class LIFLayer(nn.Module):
    """LIF a corrente istantanea, leak beta=sigmoid(w) appreso per neurone, reset hard, soglia 1."""

    adaptive = False

    def __init__(self, n_in, n_out, tau_init=2.0, in_second_moment=1.0):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        rate_aware_init_(self.fc, in_second_moment)
        self.w_beta = nn.Parameter(torch.full((n_out,), float(torch.logit(torch.tensor(1 - 1 / tau_init)))))
        self.alpha = 2.0  # pendenza surrogate, modificabile a runtime (scheduling)
        self.u = None

    def reset(self, batch, device, dones=None):
        """dones=None: azzera tutto. dones (batch,) in {0,1}: azzera solo gli env terminati."""
        if self.u is None or self.u.shape[0] != batch or dones is None:
            self.u = torch.zeros(batch, self.fc.out_features, device=device)
        else:
            self.u = self.u * (1 - dones.view(-1, 1).to(self.u.dtype))

    def detach_state(self):
        if self.u is not None:
            self.u = self.u.detach()

    def forward(self, x):
        u = torch.sigmoid(self.w_beta) * self.u + self.fc(x)
        s = SpikeFn.apply(u - 1.0, self.alpha)
        self.u = u * (1 - s)
        return s


class ALIFLayer(LIFLayer):
    """LIF con soglia adattiva: theta_t = 1 + gamma*a_t, a_t = rho*a_{t-1} + s_{t-1} (rho appreso)."""

    adaptive = True

    def __init__(self, n_in, n_out, tau_init=2.0, in_second_moment=1.0, gamma=0.5, tau_adapt_init=10.0):
        super().__init__(n_in, n_out, tau_init, in_second_moment)
        self.gamma = gamma
        self.w_rho = nn.Parameter(torch.full((n_out,), float(torch.logit(torch.tensor(1 - 1 / tau_adapt_init)))))
        self.a = None
        self.s_prev = None

    def reset(self, batch, device, dones=None):
        fresh = self.a is None or self.a.shape[0] != batch or dones is None
        super().reset(batch, device, dones)
        if fresh:
            self.a = torch.zeros(batch, self.fc.out_features, device=device)
            self.s_prev = torch.zeros_like(self.a)
        else:
            m = (1 - dones.view(-1, 1).to(self.a.dtype))
            self.a, self.s_prev = self.a * m, self.s_prev * m

    def detach_state(self):
        super().detach_state()
        if self.a is not None:
            self.a, self.s_prev = self.a.detach(), self.s_prev.detach()

    def forward(self, x):
        self.a = torch.sigmoid(self.w_rho) * self.a + self.s_prev
        u = torch.sigmoid(self.w_beta) * self.u + self.fc(x)
        s = SpikeFn.apply(u - (1.0 + self.gamma * self.a), self.alpha)
        self.u = u * (1 - s)
        self.s_prev = s
        return s
