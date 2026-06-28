"""Validate the `snek` Rust bindings: batch stepping, encoding shapes, and
JSON round-trip via the official move-request format."""

import json

import numpy as np
import snek


def test_channels_constant():
    assert snek.CHANNELS == 9


def test_batch_construction_and_shapes():
    batch = snek.GameBatch(11, 11, 2, count=8, seed=1)
    assert batch.count == 8
    assert batch.num_snakes == 2
    assert batch.width == 11 and batch.height == 11
    assert batch.channels == 9

    obs = batch.encode()
    assert obs.shape == (8, 2, 9, 21, 21)
    assert obs.dtype == np.float32
    # Board-mask channel (8) marks the real 11x11 board inside the head-centred canvas.
    assert np.all(obs[:, :, 8].sum(axis=(2, 3)) == 11 * 11)
    # Each snake sees exactly one own-head cell at the start.
    assert np.all(obs[:, :, 0].sum(axis=(2, 3)) == 1.0)
    assert np.all(obs[:, :, 0, 10, 10] == 1.0)


def test_alive_and_legal_shapes():
    batch = snek.GameBatch(11, 11, 4, count=3, seed=2)
    alive = batch.alive()
    assert alive.shape == (3, 4)
    assert np.all(alive == 1)  # everyone alive at start
    legal = batch.legal_moves()
    assert legal.shape == (3, 4, 4)
    # Coiled start (body stacked) has neck == head, so no reversal is masked yet.
    assert legal.dtype == np.uint8


def test_stepping_advances_and_terminates():
    batch = snek.GameBatch(11, 11, 2, count=4, seed=3)
    rng = np.random.default_rng(0)
    steps = 0
    while not np.all(batch.done()) and steps < 500:
        actions = rng.integers(0, 4, size=(4, 2), dtype=np.uint8)
        batch.step(actions)
        steps += 1
    assert steps < 500, "random games should terminate well before the cap"
    # Winners are -1 (ongoing/draw) or a valid snake index.
    winners = batch.winners()
    assert winners.shape == (4,)
    assert np.all((winners >= -1) & (winners < 2))


def test_reset_done_restarts_games():
    batch = snek.GameBatch(11, 11, 2, count=2, seed=4)
    rng = np.random.default_rng(1)
    while not np.all(batch.done()):
        batch.step(rng.integers(0, 4, size=(2, 2), dtype=np.uint8))
    assert batch.reset_done() == 2
    assert not np.any(batch.done())


def test_move_request_round_trip():
    # Minimal official /move payload: a length-3 snake near the centre.
    payload = {
        "turn": 5,
        "board": {
            "width": 11,
            "height": 11,
            "food": [{"x": 5, "y": 6}],
            "hazards": [],
            "snakes": [
                {
                    "id": "me",
                    "health": 90,
                    "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
                },
                {
                    "id": "opp",
                    "health": 80,
                    "body": [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 3}],
                },
            ],
        },
        "you": {"id": "me"},
    }
    obs, me, legal = snek.encode_move_request(json.dumps(payload))
    assert me == 0
    assert obs.shape == (9, 21, 21)
    # My head plane is centred in the egocentric canvas.
    assert obs[0, 10, 10] == 1.0
    # Food plane (6) marks (5,6), one cell above the centred head at (5,5).
    assert obs[6, 11, 10] == 1.0
    # Health plane (2) broadcasts 90/100.
    assert np.allclose(obs[2], 0.9)
    # Opponent head plane (3) marks (1,1), four cells down-left from my head.
    assert obs[3, 6, 6] == 1.0
    # Moving Down (toward neck at y=4) is illegal; legal is uint8 length 4.
    assert legal.shape == (4,)
    assert legal[1] == 0  # Down reverses onto neck


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
