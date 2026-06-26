"""Free-for-all (N=4) end-to-end check: the engine, equilibrium solver, and
search are all generic over snake count, so a 4-snake game must run through the
search and produce valid per-snake policies."""

import numpy as np
import snek
from azsnek.net import AZNet, NetConfig, device_auto
from azsnek.search import run_search, sample_actions


def test_ffa_search_and_play():
    device = device_auto()
    net = AZNet(NetConfig(channels=snek.CHANNELS, filters=16, blocks=2)).to(device)
    batch = snek.GameBatch(11, 11, 4, count=8, seed=5)

    policy = run_search(batch, net, device, depth=1, tau=6.0, iters=60)
    assert policy.shape == (8, 4, 4)
    sums = policy.sum(axis=2)
    assert np.all((np.abs(sums - 1.0) < 1e-3) | (sums < 1e-6))

    rng = np.random.default_rng(0)
    for _ in range(25):
        policy = run_search(batch, net, device, depth=1, tau=6.0, iters=40)
        batch.step(sample_actions(policy, rng))
    # Games make progress: at least one snake has died somewhere.
    assert int(batch.alive().sum()) < 8 * 4
