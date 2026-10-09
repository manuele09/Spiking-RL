"""Actor/critic intercambiabili per PPO: MLP (A0) e actor spiking (A2/A3)."""
import torch
import torch.nn as nn

from spiking_rl.snn import ALIFLayer, DirectEncoder, LIFLayer, PopulationEncoder


def mlp(n_in, hidden, n_out, act=nn.ELU, out_std=1.0):
    layers, d = [], n_in
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, 2 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers += [lin, act()]
        d = h
    out = nn.Linear(d, n_out)
    nn.init.orthogonal_(out.weight, out_std)
    nn.init.zeros_(out.bias)
    return nn.Sequential(*layers, out)


class MLPActor(nn.Module):
    spiking = False

    def __init__(self, n_obs, n_out, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(n_obs, hidden, n_out, out_std=0.01)

    def forward(self, obs):
        return self.net(obs)

    def reset(self, batch, device, dones=None):
        pass


class SpikingActor(nn.Module):
    """obs -> encoder -> [LIF/ALIF x L] (T passi interni) -> neuroni di uscita non-spiking.

    decoder='membrane': uscita = media su T di un readout lineare dei spike dell'ultimo strato
    (equivalente a integrare la membrana di neuroni non-spiking senza leak).
    carry=False: stato azzerato a ogni chiamata (feedforward). carry=True: stato portato tra passi.
    """

    spiking = True

    def __init__(self, n_obs, n_out, hidden=(128, 128), T=4, neuron="lif", encoding="direct", carry=False):
        super().__init__()
        self.enc = DirectEncoder(n_obs) if encoding == "direct" else PopulationEncoder(n_obs)
        cls = {"lif": LIFLayer, "alif": ALIFLayer}[neuron]
        dims = (self.enc.out_dim,) + tuple(hidden)
        self.layers = nn.ModuleList(
            cls(dims[i], dims[i + 1], in_second_moment=(self.enc.second_moment if i == 0 else 0.15))
            for i in range(len(hidden))
        )
        self.out = nn.Linear(dims[-1], n_out)
        nn.init.normal_(self.out.weight, std=0.01 / (dims[-1] * 0.15) ** 0.5)
        nn.init.zeros_(self.out.bias)
        self.T, self.carry = T, carry
        self.last_rates = [0.0] * len(self.layers)

    def reset(self, batch, device, dones=None):
        for l in self.layers:
            l.reset(batch, device, dones)

    def set_alpha(self, a):
        for l in self.layers:
            l.alpha = a

    def forward(self, obs):
        if not self.carry:
            self.reset(obs.shape[0], obs.device)
        x0 = self.enc(obs)
        acc, rates = 0.0, [0.0] * len(self.layers)
        for _ in range(self.T):
            x = x0
            for i, l in enumerate(self.layers):
                x = l(x)
                rates[i] = rates[i] + x.detach().mean() / self.T
            acc = acc + self.out(x)
        self.last_rates = [float(r) for r in rates]
        return acc / self.T


class ActorCritic(nn.Module):
    """Actor intercambiabile (ANN/SNN) + critic ANN. Azioni continue (gaussiana) o discrete (categorica)."""

    def __init__(self, n_obs, n_act, discrete, actor_kind="ann", hidden=(128, 128), critic_hidden=(128, 128),
                 T=4, neuron="lif", encoding="direct", carry=False, init_log_std=0.0):
        super().__init__()
        self.discrete = discrete
        if actor_kind == "ann":
            self.actor = MLPActor(n_obs, n_act, hidden)
        else:
            self.actor = SpikingActor(n_obs, n_act, hidden, T=T, neuron=neuron, encoding=encoding, carry=carry)
        self.critic = mlp(n_obs, critic_hidden, 1, out_std=1.0)
        if not discrete:
            self.log_std = nn.Parameter(torch.full((n_act,), float(init_log_std)))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def dist(self, obs):
        out = self.actor(obs)
        if self.discrete:
            return torch.distributions.Categorical(logits=out)
        return torch.distributions.Normal(out, self.log_std.exp().expand_as(out))

    def act(self, obs, action=None, deterministic=False):
        d = self.dist(obs)
        if action is None:
            if deterministic:
                action = d.logits.argmax(-1) if self.discrete else d.mean
            else:
                action = d.sample()
        lp = d.log_prob(action)
        if not self.discrete:
            lp = lp.sum(-1)
            ent = d.entropy().sum(-1)
        else:
            ent = d.entropy()
        return action, lp, ent
