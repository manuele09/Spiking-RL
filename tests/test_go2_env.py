import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
import spiking_rl.envs  # noqa: F401


def test_obs_and_step_shapes():
    e = gym.make("Go2Balance-v0")
    o, _ = e.reset(seed=0)
    assert o.shape == (45,) and np.isfinite(o).all()
    o, r, te, tr, info = e.step(np.zeros(12))
    assert o.shape == (45,) and np.isfinite(r)


def test_zero_action_survives_without_pushes():
    e = gym.make("Go2Balance-v0", push=False)
    e.reset(seed=1)
    for _ in range(100):
        _, _, te, tr, _ = e.step(np.zeros(12))
        assert not te
    assert e.unwrapped.d.qpos[2] > 0.2


def test_fall_terminates():
    e = gym.make("Go2Balance-v0", push=False, randomize=False)
    e.reset(seed=0)
    term = False
    for _ in range(100):
        _, _, term, _, _ = e.step(np.full(12, -4.0))  # azioni estreme: il robot collassa
        if term:
            break
    assert term
