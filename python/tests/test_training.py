"""Smoke-test the self-play -> train -> evaluate pipeline end to end (tiny)."""

import snek
import torch
from azsnek.evaluate import evaluate
from azsnek.net import AZNet, NetConfig
from azsnek.selfplay import SelfPlayConfig, generate
from azsnek.train import train_on_samples


def test_generate_train_eval_pipeline():
    device = torch.device("cpu")
    net = AZNet(NetConfig(channels=snek.CHANNELS, filters=16, blocks=2)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    cfg = SelfPlayConfig(count=8, sims=8, samples_per_gen=200, eval_batch_size=256)
    samples = generate(net, device, cfg, seed=0)
    assert samples.obs.shape[0] >= 200
    assert samples.obs.shape[1:] == (snek.CHANNELS, 21, 21)
    assert samples.pol.shape[1] == 4
    # Policy targets are valid distributions (rows for alive snakes sum to ~1).
    sums = samples.pol.sum(axis=1)
    assert ((abs(sums - 1.0) < 1e-3) | (sums < 1e-6)).all()
    # Value targets are in {-1, 0, 1}.
    assert set(samples.z.tolist()) <= {-1.0, 0.0, 1.0}

    losses = train_on_samples(net, opt, samples, device, steps=5, batch_size=64)
    assert losses["policy_loss"] >= 0.0
    assert losses["value_loss"] >= 0.0

    res = evaluate(net, device, games=8, sims=8, eval_batch_size=256)
    assert 0.0 <= res["win_rate"] <= 1.0
    assert res["wins"] + res["losses"] + res["draws"] + res["unfinished"] == 8


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
