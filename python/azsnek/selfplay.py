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
from .search import run_search, sample_actions


@dataclass
class SelfPlayConfig:
    board: int = 11
    num_snakes: int = 2
    count: int = 32  # parallel games
    depth: int = 2
    tau: float = 30.0
    iters: int = 120
    eval_batch_size: int = 8192
    samples_per_gen: int = 20_000
    max_turns: int = 0  # 0 = play until terminal; positive values cap games as draws
    dirichlet_frac: float = 0.25  # root exploration noise mix (0 disables)
    dirichlet_alpha: float = 0.3
    # Value-target shaping (Albatross-style): discounted TD(lambda) return of a
    # dense per-step reward, instead of the undiscounted terminal win/loss.
    gamma: float = 0.97  # discount; <1 makes surviving longer worth more
    lam: float = 0.5  # TD(lambda): 0 = pure bootstrap, 1 = discounted Monte-Carlo
    living_reward: float = 0.01  # per-step reward for staying alive
    terminal_reward: float = 1.0  # +win / -death at the end
    exploration_prob: float = 0.25  # mix this much uniform-legal into the play policy


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
    pol: list = field(default_factory=list)  # each [N, 4]
    alive: list = field(default_factory=list)  # each [N] bool
    val: list = field(default_factory=list)  # each [N] root bootstrap value
    reward: list = field(default_factory=list)  # each [N] step reward
    turns: int = 0


def _step_reward(
    alive_before: np.ndarray,  # [N] bool, alive at this state
    alive_after: np.ndarray,  # [N] bool, alive after the step
    ended: bool,
    winner: int,  # winner index, or -1 for draw / not-ended
    living: float,
    terminal: float,
) -> np.ndarray:
    """Dense per-snake reward for the transition out of this state."""
    n = alive_before.shape[0]
    r = np.zeros(n, dtype=np.float32)
    for p in range(n):
        if not alive_before[p]:
            continue
        if not alive_after[p]:  # died this step
            r[p] = -terminal
        elif ended and winner == p:  # last snake standing
            r[p] = terminal
        elif ended:  # game ended but p didn't win outright (draw / turn cap)
            r[p] = 0.0
        else:  # survived, game continues
            r[p] = living
    return r


def _lambda_returns(slot: _Slot, player: int, gamma: float, lam: float) -> tuple[list[int], np.ndarray]:
    """TD(lambda) return target for `player` over the steps it was alive.

    Snakes never resurrect, so a player's live steps are a contiguous prefix.
    Returns (live_step_indices, targets) with targets clipped to [-1, 1] to match
    the tanh value head."""
    steps = [t for t in range(len(slot.alive)) if slot.alive[t][player]]
    if not steps:
        return steps, np.zeros(0, dtype=np.float32)
    targets = np.zeros(len(steps), dtype=np.float32)
    g_next = 0.0
    for i in range(len(steps) - 1, -1, -1):
        t = steps[i]
        r_t = float(slot.reward[t][player])
        if i == len(steps) - 1:  # player's terminal step: no bootstrap
            g = r_t
        else:
            v_next = float(slot.val[steps[i + 1]][player])
            g = r_t + gamma * ((1.0 - lam) * v_next + lam * g_next)
        g_next = g
        targets[i] = g
    np.clip(targets, -1.0, 1.0, out=targets)
    return steps, targets


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
        policy, root_vals = run_search(
            batch, net, device, cfg.depth, cfg.tau, cfg.iters, cfg.eval_batch_size,
            return_root_values=True,
        )  # policy [count, N, 4], root_vals [count, N]
        obs = batch.encode()  # [count, N, C, H, W]
        alive = batch.alive().astype(bool)  # [count, N], alive at this state
        turns_total += int(np.sum(batch.done() == 0))  # games still live this step
        for g in range(cfg.count):
            slots[g].obs.append(obs[g])
            slots[g].pol.append(policy[g])
            slots[g].alive.append(alive[g])
            slots[g].val.append(root_vals[g])
            slots[g].turns += 1

        # Explore: mix Dirichlet noise + uniform-legal into the *played* action
        # (the stored policy target stays the clean search policy).
        play_policy = add_root_noise(policy, rng, cfg.dirichlet_frac, cfg.dirichlet_alpha)
        play_policy = mix_uniform(play_policy, cfg.exploration_prob)
        actions = sample_actions(play_policy, rng)
        batch.step(actions)
        done = batch.done()
        winners = batch.winners()
        alive_after = batch.alive().astype(bool)  # [count, N], after the step

        for g in range(cfg.count):
            overrun = cfg.max_turns > 0 and slots[g].turns >= cfg.max_turns
            ended = bool(done[g]) or overrun
            slots[g].reward.append(
                _step_reward(alive[g], alive_after[g], ended, int(winners[g]),
                             cfg.living_reward, cfg.terminal_reward)
            )
            if not ended:
                continue
            # Finalize: per-player discounted TD(lambda) return over its live steps.
            for p in range(cfg.num_snakes):
                steps, targets = _lambda_returns(slots[g], p, cfg.gamma, cfg.lam)
                if not steps:
                    continue
                out_obs.append(np.stack([slots[g].obs[t][p] for t in steps]))
                out_pol.append(np.stack([slots[g].pol[t][p] for t in steps]))
                out_z.append(targets)
                collected += len(steps)
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
