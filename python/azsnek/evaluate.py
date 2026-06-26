"""Evaluate the search agent against the flood-fill baseline opponent."""

from __future__ import annotations

import numpy as np
import snek
import torch

from .net import AZNet
from .search import greedy_actions, run_search


@torch.no_grad()
def evaluate(
    net: AZNet,
    device: torch.device,
    board: int = 11,
    games: int = 200,
    depth: int = 2,
    tau: float = 6.0,
    iters: int = 120,
    seed: int = 12345,
) -> dict:
    """Snake 0 = our search agent (greedy), snake 1 = flood-fill baseline.

    Returns a dict with win/loss/draw counts and the agent win rate (draws count
    as half), over `games` parallel duels.
    """
    batch = snek.GameBatch(board, board, 2, count=games, seed=seed)
    rng = np.random.default_rng(seed)

    steps = 0
    while not np.all(batch.done()) and steps < 2 * board * board:
        policy = run_search(batch, net, device, depth, tau, iters)
        agent_act = greedy_actions(policy)[:, 0]
        base_act = batch.baseline_actions()[:, 1]
        actions = np.stack([agent_act, base_act], axis=1).astype(np.uint8)
        batch.step(actions)
        steps += 1
        _ = rng  # reserved for optional stochastic play

    winners = batch.winners()
    done = batch.done().astype(bool)
    wins = int(np.sum(winners == 0))
    losses = int(np.sum(winners == 1))
    draws = int(np.sum(done & (winners == -1)))
    unfinished = int(np.sum(~done))
    decided = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / decided if decided else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "unfinished": unfinished,
        "win_rate": win_rate,
    }
