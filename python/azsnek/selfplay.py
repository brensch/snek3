"""Self-play data generation.

Runs many games in parallel; at each turn the equilibrium search produces a
policy target per snake, we sample an action from it, and when a game ends we
back-fill the value target (game outcome from each snake's perspective) onto all
that game's recorded positions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import snek
import torch

from .net import AZNet
from .search import mcts_search, sample_actions


@dataclass
class SelfPlayConfig:
    board: int = 11
    num_snakes: int = 2
    count: int = 32  # parallel games
    sims: int = 128  # MCTS simulations per move
    c_puct: float = 1.5  # PUCT exploration constant
    eval_batch_size: int = 8192
    samples_per_gen: int = 20_000
    max_turns: int = 0  # 0 = play until terminal; positive values cap games as draws
    dirichlet_frac: float = 0.25  # root exploration noise mix (0 disables)
    dirichlet_alpha: float = 0.3
    exploration_prob: float = 0.25  # mix this much uniform-legal into the played action


@dataclass
class Samples:
    obs: np.ndarray  # [K, C, H, W] float32
    pol: np.ndarray  # [K, 4] float32
    z: np.ndarray  # [K] float32
    turns: int = 0  # total board-turns stepped (for throughput)
    games: int = 0  # total games finished


class ReplayBuffer:
    """Sliding window of recent self-play samples (AlphaZero trains on a window
    of recent games, not just the latest generation)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._obs: deque[np.ndarray] = deque()
        self._pol: deque[np.ndarray] = deque()
        self._z: deque[np.ndarray] = deque()
        self._n = 0

    def add(self, s: Samples) -> None:
        self._obs.append(s.obs)
        self._pol.append(s.pol)
        self._z.append(s.z)
        self._n += len(s.z)
        while self._n > self.capacity and len(self._z) > 1:
            self._n -= len(self._z[0])
            self._obs.popleft()
            self._pol.popleft()
            self._z.popleft()

    def __len__(self) -> int:
        return self._n

    def dataset(self) -> Samples:
        return Samples(
            obs=np.concatenate(self._obs, axis=0),
            pol=np.concatenate(self._pol, axis=0),
            z=np.concatenate(self._z, axis=0),
        )


def add_root_noise(policy: np.ndarray, rng: np.random.Generator, frac: float, alpha: float) -> np.ndarray:
    """Mix Dirichlet noise into the root policy over each agent's *legal* moves
    (the nonzero entries), per AlphaZero, to keep self-play exploring."""
    if frac <= 0:
        return policy
    out = policy.copy().reshape(-1, policy.shape[-1])
    for row in out:
        legal = row > 0
        k = int(legal.sum())
        if k >= 2:
            row[legal] = (1 - frac) * row[legal] + frac * rng.dirichlet([alpha] * k)
    return out.reshape(policy.shape)


def mix_uniform(policy: np.ndarray, frac: float) -> np.ndarray:
    """Mix `frac` of a uniform-over-legal distribution into the play policy, to
    keep self-play exploring and produce more decisive, varied games."""
    if frac <= 0:
        return policy
    out = policy.copy().reshape(-1, policy.shape[-1])
    for row in out:
        legal = row > 0
        k = int(legal.sum())
        if k >= 1:
            row[legal] = (1 - frac) * row[legal] + frac * (1.0 / k)
    return out.reshape(policy.shape)


@dataclass
class _Slot:
    """Pending records for one parallel game until it finishes."""

    obs: list = field(default_factory=list)  # each [N, C, H, W]
    pol: list = field(default_factory=list)  # each [N, 4] MCTS visit-count policy
    alive: list = field(default_factory=list)  # each [N] bool
    turns: int = 0


def _outcome(winner: int, n: int) -> np.ndarray:
    """Undiscounted value target per snake: +1 winner, -1 loser, 0 draw."""
    if winner < 0:
        return np.zeros(n, dtype=np.float32)
    z = -np.ones(n, dtype=np.float32)
    z[winner] = 1.0
    return z


@torch.no_grad()
def generate(net: AZNet, device: torch.device, cfg: SelfPlayConfig, seed: int) -> Samples:
    """AlphaZero self-play: MCTS produces a visit-count policy target per move;
    the value target is the (undiscounted) game outcome from each snake's
    perspective, back-filled when the game ends."""
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
        policy, _root_vals = mcts_search(
            batch, net, device, sims=cfg.sims, c_puct=cfg.c_puct,
            eval_batch_size=cfg.eval_batch_size,
        )  # policy [count, N, 4] = root visit-count distribution
        obs = batch.encode()  # [count, N, C, H, W]
        alive = batch.alive().astype(bool)  # [count, N]
        turns_total += int(np.sum(batch.done() == 0))
        for g in range(cfg.count):
            slots[g].obs.append(obs[g])
            slots[g].pol.append(policy[g])
            slots[g].alive.append(alive[g])
            slots[g].turns += 1

        # Explore: mix Dirichlet + uniform-legal into the *played* action; the
        # stored policy target stays the clean MCTS visit-count distribution.
        play_policy = add_root_noise(policy, rng, cfg.dirichlet_frac, cfg.dirichlet_alpha)
        play_policy = mix_uniform(play_policy, cfg.exploration_prob)
        actions = sample_actions(play_policy, rng)
        batch.step(actions)
        done = batch.done()
        winners = batch.winners()

        for g in range(cfg.count):
            overrun = cfg.max_turns > 0 and slots[g].turns >= cfg.max_turns
            if not (bool(done[g]) or overrun):
                continue
            z = _outcome(int(winners[g]), cfg.num_snakes)
            for rec_obs, rec_pol, rec_alive in zip(slots[g].obs, slots[g].pol, slots[g].alive):
                live = rec_alive
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
