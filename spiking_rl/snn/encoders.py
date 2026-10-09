"""Encoder per osservazioni continue -> ingresso del primo strato spiking."""
import torch
import torch.nn as nn


class DirectEncoder(nn.Module):
    """Encoding diretto: l'osservazione (normalizzata) e' iniettata come corrente a ogni passo interno."""

    def __init__(self, n_obs):
        super().__init__()
        self.out_dim = n_obs
        self.second_moment = 1.0

    def forward(self, obs):
        return obs


class PopulationEncoder(nn.Module):
    """Population coding con campi recettivi gaussiani (stile PopSAN), centri e larghezze appresi.

    Restituisce l'attivazione analogica dei campi (valori in [0,1]) iniettata come corrente.
    """

    def __init__(self, n_obs, pop=10, lo=-3.0, hi=3.0):
        super().__init__()
        centers = torch.linspace(lo, hi, pop).repeat(n_obs, 1)  # (n_obs, pop)
        self.centers = nn.Parameter(centers)
        self.log_sigma = nn.Parameter(torch.full_like(centers, float(torch.log(torch.tensor((hi - lo) / pop)))))
        self.out_dim = n_obs * pop
        self.second_moment = 0.15

    def forward(self, obs):
        z = (obs.unsqueeze(-1) - self.centers) / self.log_sigma.exp()
        return torch.exp(-0.5 * z ** 2).flatten(1)
