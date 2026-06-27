"""Evaluate the search agent against the flood-fill baseline opponent."""

from __future__ import annotations

import numpy as np
import snek
import torch

from .net import AZNet
from .search import greedy_actions, mcts_search


@torch.no_grad()
def evaluate(
    net: AZNet,
    device: torch.device,
    board: int = 11,
    games: int = 200,
    sims: int = 128,
    c_puct: float = 1.5,
    eval_batch_size: int = 8192,
    max_turns: int = 0,
    seed: int = 12345,
) -> dict:
    """Snake 0 = our MCTS agent (greedy / most-visited), snake 1 = flood-fill
    baseline. Returns win/loss/draw counts and the agent win rate (draws = half)
    over `games` parallel duels."""
    batch = snek.GameBatch(board, board, 2, count=games, seed=seed)

    steps = 0
    while not np.all(batch.done()) and (max_turns <= 0 or steps < max_turns):
        policy, _ = mcts_search(batch, net, device, sims=sims, c_puct=c_puct,
                                eval_batch_size=eval_batch_size)
        agent_act = greedy_actions(policy)[:, 0]
        base_act = batch.baseline_actions()[:, 1]
        actions = np.stack([agent_act, base_act], axis=1).astype(np.uint8)
        batch.step(actions)
        steps += 1

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


@torch.no_grad()
def evaluate_vs_net(
    net: AZNet,
    opponent: AZNet,
    device: torch.device,
    board: int = 11,
    games: int = 64,
    sims: int = 128,
    c_puct: float = 1.5,
    eval_batch_size: int = 8192,
    max_turns: int = 0,
    seed: int = 999,
) -> float:
    """Head-to-head win rate of `net` (snake 0) vs `opponent` (snake 1).

    Both pick greedily from their own MCTS. Draws count as half. A *relative*
    skill signal that isolates net improvement from the fixed flood-fill
    baseline. Returns win_rate for `net` in [0, 1].
    """
    batch = snek.GameBatch(board, board, 2, count=games, seed=seed)
    steps = 0
    while not np.all(batch.done()) and (max_turns <= 0 or steps < max_turns):
        pol_a, _ = mcts_search(batch, net, device, sims=sims, c_puct=c_puct, eval_batch_size=eval_batch_size)
        pol_b, _ = mcts_search(batch, opponent, device, sims=sims, c_puct=c_puct, eval_batch_size=eval_batch_size)
        actions = np.stack([greedy_actions(pol_a)[:, 0], greedy_actions(pol_b)[:, 1]], axis=1).astype(np.uint8)
        batch.step(actions)
        steps += 1
    winners = batch.winners()
    done = batch.done().astype(bool)
    wins = int(np.sum(winners == 0))
    losses = int(np.sum(winners == 1))
    draws = int(np.sum(done & (winners == -1)))
    decided = wins + losses + draws
    return (wins + 0.5 * draws) / decided if decided else 0.0
