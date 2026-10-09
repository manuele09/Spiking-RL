import torch

from spiking_rl.models import ActorCritic, SpikingActor
from spiking_rl.snn import ALIFLayer, LIFLayer, PopulationEncoder


def test_lif_rates_and_grad():
    torch.manual_seed(0)
    net = SpikingActor(45, 12, (256, 256), T=4)
    out = net(torch.randn(512, 45))
    out.pow(2).mean().backward()
    assert all(0.05 < r < 0.6 for r in net.last_rates), net.last_rates
    g = net.layers[0].fc.weight.grad.norm().item()
    assert 0 < g < 1e3


def test_reset_by_dones():
    for cls in (LIFLayer, ALIFLayer):
        l = cls(8, 16)
        l.reset(4, "cpu")
        l(torch.randn(4, 8) * 3)
        l.reset(4, "cpu", dones=torch.tensor([1.0, 0.0, 1.0, 0.0]))
        assert l.u[0].abs().sum() == 0 and l.u[2].abs().sum() == 0
        assert l.u[1].abs().sum() > 0 or l.u[3].abs().sum() > 0


def test_carry_keeps_state():
    net = SpikingActor(5, 2, (16,), T=2, carry=True)
    net.reset(3, "cpu")
    x = torch.randn(3, 5)
    a, b = net(x), net(x)
    assert not torch.allclose(a, b)  # lo stato di membrana cambia l'uscita


def test_alif_threshold_adapts():
    l = ALIFLayer(4, 8)
    l.reset(2, "cpu")
    for _ in range(5):
        l(torch.ones(2, 4) * 5)
    assert l.a.sum() > 0


def test_population_encoder_shape():
    assert PopulationEncoder(3, pop=10)(torch.randn(7, 3)).shape == (7, 30)


def test_actor_critic_both_kinds():
    for kind in ("ann", "snn"):
        for discrete in (True, False):
            m = ActorCritic(4, 2, discrete, kind, hidden=(16,), T=2)
            a, lp, ent = m.act(torch.randn(5, 4))
            assert lp.shape == (5,) and ent.shape == (5,) and m.value(torch.randn(5, 4)).shape == (5,)
