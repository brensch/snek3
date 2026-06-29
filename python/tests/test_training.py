"""Smoke-test the self-play -> train -> evaluate pipeline end to end (tiny)."""

import snek
import numpy as np
import torch
from azsnek.evaluate import evaluate
from azsnek.net import AZNet, NetConfig
from azsnek.selfplay import SelfPlayConfig, _outcome, generate
from azsnek.train import train_on_samples


def test_generate_train_eval_pipeline():
    device = torch.device("cpu")
    net = AZNet(NetConfig(channels=snek.CHANNELS, filters=16, blocks=2)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    cfg = SelfPlayConfig(count=8, sims=8, samples_per_gen=200, eval_batch_size=256)
    samples = generate(net, device, cfg, seed=0)
    assert samples.obs.shape[0] >= 200
    assert samples.obs.shape[1:] == (snek.CHANNELS, cfg.board, cfg.board)
    assert samples.pol.shape[1] == 4
    # Policy targets are valid distributions (rows for alive snakes sum to ~1).
    sums = samples.pol.sum(axis=1)
    assert ((abs(sums - 1.0) < 1e-3) | (sums < 1e-6)).all()
    # Value targets are in {-1, draw_value, 1}.
    valid_z = (
        np.isclose(samples.z, -1.0, atol=1e-4)
        | np.isclose(samples.z, cfg.draw_value, atol=1e-4)
        | np.isclose(samples.z, 1.0, atol=1e-4)
    )
    assert valid_z.all()

    losses = train_on_samples(net, opt, samples, device, steps=5, batch_size=64)
    assert losses["policy_loss"] >= 0.0
    assert losses["value_loss"] >= 0.0

    res = evaluate(net, device, games=8, sims=8, eval_batch_size=256)
    assert 0.0 <= res["win_rate"] <= 1.0
    assert res["wins"] + res["losses"] + res["draws"] + res["unfinished"] == 8


def test_terminal_draw_targets_only_final_survivors():
    z = _outcome(-1, 4, final_alive=[False, True, True, False], draw_value=-0.25)
    assert z.tolist() == [-1.0, -0.25, -0.25, -1.0]

    z = _outcome(2, 4, final_alive=[False, True, True, False], draw_value=-0.25)
    assert z.tolist() == [-1.0, -1.0, 1.0, -1.0]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
