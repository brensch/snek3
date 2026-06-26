"""Network forward pass + end-to-end search bridge on whatever device is available."""

import numpy as np
import snek
import torch
from azsnek.net import AZNet, NetConfig, device_auto
from azsnek.search import greedy_actions, run_search, sample_actions


def test_net_forward_shapes():
    net = AZNet(NetConfig(channels=snek.CHANNELS))
    x = torch.zeros(5, snek.CHANNELS, 11, 11)
    logits, value = net(x)
    assert logits.shape == (5, 4)
    assert value.shape == (5,)
    assert torch.all(value.abs() <= 1.0)


def test_net_infer_masks_illegal():
    net = AZNet(NetConfig(channels=snek.CHANNELS))
    x = torch.zeros(2, snek.CHANNELS, 11, 11)
    mask = torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.uint8)
    probs, _ = net.infer(x, mask)
    # Masked actions get zero probability; rows sum to 1.
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert probs[0, 1] == 0.0 and probs[0, 3] == 0.0


def test_full_search_step_produces_valid_policy():
    device = device_auto()
    net = AZNet(NetConfig(channels=snek.CHANNELS)).to(device)
    batch = snek.GameBatch(11, 11, 2, count=16, seed=7)

    policy = run_search(batch, net, device, depth=2, tau=6.0, iters=100)
    assert policy.shape == (16, 2, 4)
    assert policy.dtype == np.float32
    # Each alive snake's policy is a valid distribution.
    sums = policy.sum(axis=2)
    assert np.all((np.abs(sums - 1.0) < 1e-3) | (sums < 1e-6))
    assert np.all(policy >= 0.0)


def test_search_drives_a_full_self_play_game():
    device = device_auto()
    net = AZNet(NetConfig(channels=snek.CHANNELS)).to(device)
    batch = snek.GameBatch(11, 11, 2, count=8, seed=11)
    rng = np.random.default_rng(0)

    steps = 0
    while not np.all(batch.done()) and steps < 400:
        policy = run_search(batch, net, device, depth=2, tau=6.0, iters=60)
        actions = sample_actions(policy, rng)
        batch.step(actions)
        steps += 1
    assert steps < 400, "untrained search games should still terminate"
    # greedy_actions also works on the final policy without error.
    _ = greedy_actions(run_search(batch, net, device, depth=1, tau=6.0, iters=30))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
