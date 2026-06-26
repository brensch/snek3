"""Self-play data generation.

Runs many games in parallel; at each turn the equilibrium search produces a
policy target per snake, we sample an action from it, and when a game ends we
back-fill the value target (game outcome from each snake's perspective) onto all
that game's recorded positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import snek
import torch

from .net import AZNet
from .search import run_search, sample_actions


@dataclass
class SelfPlayConfig:
    board: int = 11
    num_snakes: int = 2
    count: int = 128  # parallel games
    depth: int = 2
    tau: float = 6.0
    iters: int = 120
    samples_per_gen: int = 20_000
    max_turns: int = 400  # safety cap per game


@dataclass
class Samples:
    obs: np.ndarray  # [K, C, H, W] float32
    pol: np.ndarray  # [K, 4] float32
    z: np.ndarray  # [K] float32
    turns: int = 0  # total board-turns stepped (for throughput)
    games: int = 0  # total games finished


@dataclass
class _Slot:
    """Pending records for one parallel game until it finishes."""

    obs: list = field(default_factory=list)  # each [N, C, H, W]
    pol: list = field(default_factory=list)  # each [N, 4]
    alive: list = field(default_factory=list)  # each [N]
    turns: int = 0


def _outcome(winner: int, n: int) -> np.ndarray:
    """Value target per snake: +1 winner, -1 loser, 0 draw (winner == -1)."""
    if winner < 0:
        return np.zeros(n, dtype=np.float32)
    z = -np.ones(n, dtype=np.float32)
    z[winner] = 1.0
    return z


@torch.no_grad()
def generate(net: AZNet, device: torch.device, cfg: SelfPlayConfig, seed: int) -> Samples:
    rng = np.random.default_rng(seed)
    batch = snek.GameBatch(cfg.board, cfg.board, cfg.num_snakes, count=cfg.count, seed=seed)
    slots = [_Slot() for _ in range(cfg.count)]

    out_obs: list[np.ndarray] = []
    out_pol: list[np.ndarray] = []
    out_z: list[np.ndarray] = []
    collected = 0
    turns_total = 0
    games_total = 0

    while collected < cfg.samples_per_gen:
        policy = run_search(batch, net, device, cfg.depth, cfg.tau, cfg.iters)
        obs = batch.encode()  # [count, N, C, H, W]
        alive = batch.alive()  # [count, N]
        turns_total += int(np.sum(batch.done() == 0))  # games still live this step
        for g in range(cfg.count):
            slots[g].obs.append(obs[g])
            slots[g].pol.append(policy[g])
            slots[g].alive.append(alive[g])
            slots[g].turns += 1

        actions = sample_actions(policy, rng)
        # Force-terminate games that overrun the turn cap so slots flush.
        batch.step(actions)
        done = batch.done()
        winners = batch.winners()

        for g in range(cfg.count):
            overrun = slots[g].turns >= cfg.max_turns
            if not (done[g] or overrun):
                continue
            z = _outcome(int(winners[g]), cfg.num_snakes)
            for rec_obs, rec_pol, rec_alive in zip(slots[g].obs, slots[g].pol, slots[g].alive):
                live = rec_alive.astype(bool)
                if not live.any():
                    continue
                out_obs.append(rec_obs[live])
                out_pol.append(rec_pol[live])
                out_z.append(z[live])
                collected += int(live.sum())
            slots[g] = _Slot()
            games_total += 1

        batch.reset_done()

    return Samples(
        obs=np.concatenate(out_obs, axis=0),
        pol=np.concatenate(out_pol, axis=0),
        z=np.concatenate(out_z, axis=0),
        turns=turns_total,
        games=games_total,
    )
