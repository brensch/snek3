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
from .search import mcts_search, perf_reset, perf_snapshot, run_search, run_search_hetero, sample_actions


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
    # Albatross equilibrium-search params (proxy/response).
    depth: int = 2  # fixed-depth equilibrium search plies
    iters: int = 120  # logit-equilibrium SFP iterations per node
    tau_min: float = 0.5  # proxy: low end of the per-episode temperature distribution
    tau_max: float = 10.0  # proxy: high end (>= ~10 ~ near-optimal play)
    response_tau: float = 12.0  # response: the rational agent's fixed temperature (tau_R)
    draw_value: float = -0.9  # terminal value of a draw (negative kills mutual-suicide draws)


@dataclass
class Samples:
    obs: np.ndarray  # [K, C, H, W] float32
    pol: np.ndarray  # [K, 4] float32
    z: np.ndarray  # [K] float32
    temp: np.ndarray | None = None  # [K] float32 per-sample temperature (Albatross)
    turns: int = 0  # total board-turns stepped (for throughput)
    games: int = 0  # total games finished
    draws: int = 0  # of `games`, how many ended without a winner (draw/timeout)
    fwd_seconds: float = 0.0  # time in net forward (GPU)
    search_seconds: float = 0.0  # time in Rust tree-build + equilibrium backup (CPU)
    inferences: int = 0  # leaf positions evaluated by the net


class ReplayBuffer:
    """Sliding window of recent self-play samples (AlphaZero trains on a window
    of recent games, not just the latest generation). Optionally carries a
    per-sample temperature (`temp`) for Albatross temperature-conditioned nets."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._obs: deque[np.ndarray] = deque()
        self._pol: deque[np.ndarray] = deque()
        self._z: deque[np.ndarray] = deque()
        self._temp: deque[np.ndarray] = deque()
        self._has_temp = False
        self._n = 0

    def add(self, s: Samples) -> None:
        self._obs.append(s.obs)
        self._pol.append(s.pol)
        self._z.append(s.z)
        if s.temp is not None:
            self._temp.append(s.temp)
            self._has_temp = True
        self._n += len(s.z)
        while self._n > self.capacity and len(self._z) > 1:
            self._n -= len(self._z[0])
            self._obs.popleft()
            self._pol.popleft()
            self._z.popleft()
            if self._has_temp:
                self._temp.popleft()

    def __len__(self) -> int:
        return self._n

    def dataset(self) -> Samples:
        return Samples(
            obs=np.concatenate(self._obs, axis=0),
            pol=np.concatenate(self._pol, axis=0),
            z=np.concatenate(self._z, axis=0),
            temp=np.concatenate(self._temp, axis=0) if self._has_temp else None,
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


@torch.no_grad()
def generate_proxy(net: AZNet, device: torch.device, cfg: SelfPlayConfig, seed: int,
                   progress_cb=None) -> Samples:
    """Albatross PROXY self-play. A per-generation temperature `tau` is sampled
    from [tau_min, tau_max]; all agents play the logit equilibrium at `tau`
    (fixed-depth equilibrium search), and the net is conditioned on `tau`. Both
    targets come from the equilibrium, not the game outcome:
      * policy target  = LE root policy at `tau`
      * value target   = LE root expected utility at `tau` (per-step bootstrap)
    so the net learns pi_P(o, tau) and v_P(o, tau). Over generations `tau` ranges
    across the interval, training the temperature-conditioned proxy.
    """
    rng = np.random.default_rng(seed)
    perf_reset()  # isolate this proxy self-play's GPU/CPU timing
    batch = snek.GameBatch(cfg.board, cfg.board, cfg.num_snakes, count=cfg.count, seed=seed)
    tau = float(rng.uniform(cfg.tau_min, cfg.tau_max))  # per-generation temperature

    out_obs: list[np.ndarray] = []
    out_pol: list[np.ndarray] = []
    out_z: list[np.ndarray] = []
    out_temp: list[np.ndarray] = []
    collected = 0
    turns_total = 0
    games_total = 0
    draws_total = 0
    next_report = 0.25

    while collected < cfg.samples_per_gen:
        if progress_cb is not None and collected >= next_report * cfg.samples_per_gen:
            progress_cb(collected, cfg.samples_per_gen)
            next_report += 0.25
        policy, root_vals = run_search(
            batch, net, device, cfg.depth, tau, cfg.iters,
            cfg.eval_batch_size, return_root_values=True, temp=tau,
            draw_value=cfg.draw_value,
        )  # policy [count, N, 4] LE policy; root_vals [count, N] LE expected utility
        obs = batch.encode()
        alive = batch.alive().astype(bool)
        turns_total += int(np.sum(batch.done() == 0))
        for g in range(cfg.count):
            live = alive[g]
            if not live.any():
                continue
            out_obs.append(obs[g][live])
            out_pol.append(policy[g][live])
            out_z.append(root_vals[g][live].astype(np.float32))
            out_temp.append(np.full(int(live.sum()), tau, dtype=np.float32))
            collected += int(live.sum())

        play = mix_uniform(policy, cfg.exploration_prob)
        actions = sample_actions(play, rng)
        batch.step(actions)
        done = batch.done().astype(bool)
        if done.any():
            w = batch.winners()
            games_total += int(done.sum())
            draws_total += int(np.sum(done & (w == -1)))
        batch.reset_done()

    perf = perf_snapshot()
    return Samples(
        obs=np.concatenate(out_obs, axis=0),
        pol=np.concatenate(out_pol, axis=0),
        z=np.concatenate(out_z, axis=0),
        temp=np.concatenate(out_temp, axis=0),
        turns=turns_total,
        games=games_total,
        draws=draws_total,
        fwd_seconds=perf["fwd_s"],
        search_seconds=perf["search_s"],
        inferences=perf["infer"],
    )


@torch.no_grad()
def generate_response(
    net: AZNet,
    proxy: AZNet,
    device: torch.device,
    cfg: SelfPlayConfig,
    seed: int,
    progress_cb=None,
) -> Samples:
    """Albatross RESPONSE self-play (2-player). Agent 0 is the rational responder
    at fixed temperature `response_tau` (tau_R); agent 1 is a weak opponent at a
    per-generation sampled temperature `tau_opp`. A heterogeneous-temperature
    equilibrium search ([tau_R, tau_opp]) over leaf values from the frozen
    `proxy` net yields, for the responder, the smooth-best-response policy and
    its expected value -- the targets for the response net, conditioned on the
    OPPONENT's temperature `tau_opp`. Both agents are driven by the same search
    (responder samples its policy, opponent samples its LE policy).
    """
    assert cfg.num_snakes == 2, "response model is defined for 2-player duels"
    rng = np.random.default_rng(seed)
    batch = snek.GameBatch(cfg.board, cfg.board, 2, count=cfg.count, seed=seed)
    tau_opp = float(rng.uniform(cfg.tau_min, cfg.tau_max))  # weak opponent temperature
    tau_pair = [cfg.response_tau, tau_opp]

    out_obs: list[np.ndarray] = []
    out_pol: list[np.ndarray] = []
    out_z: list[np.ndarray] = []
    out_temp: list[np.ndarray] = []
    collected = 0
    turns_total = 0
    games_total = 0
    next_report = 0.25

    while collected < cfg.samples_per_gen:
        if progress_cb is not None and collected >= next_report * cfg.samples_per_gen:
            progress_cb(collected, cfg.samples_per_gen)
            next_report += 0.25
        # Leaf values from the frozen proxy, conditioned on the opponent's tau
        # (the weak agent sets the game's character).
        policy, root_vals = run_search_hetero(
            batch, proxy, device, cfg.depth, tau_pair, cfg.iters,
            cfg.eval_batch_size, temp=tau_opp, draw_value=cfg.draw_value,
        )  # policy [count, 2, 4]; root_vals [count, 2]
        obs = batch.encode()
        alive = batch.alive().astype(bool)
        turns_total += int(np.sum(batch.done() == 0))
        # Record ONLY the responder (agent 0); it is what the response net learns.
        for g in range(cfg.count):
            if alive[g, 0]:
                out_obs.append(obs[g, 0][None])
                out_pol.append(policy[g, 0][None])
                out_z.append(root_vals[g, 0][None].astype(np.float32))
                out_temp.append(np.full(1, tau_opp, dtype=np.float32))
                collected += 1

        play = mix_uniform(policy, cfg.exploration_prob)
        actions = sample_actions(play, rng)
        batch.step(actions)
        games_total += int(np.sum(batch.done()))
        batch.reset_done()

    return Samples(
        obs=np.concatenate(out_obs, axis=0),
        pol=np.concatenate(out_pol, axis=0),
        z=np.concatenate(out_z, axis=0),
        temp=np.concatenate(out_temp, axis=0),
        turns=turns_total,
        games=games_total,
    )
