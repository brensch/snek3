"""Record full self-play / evaluation games as replayable frame sequences for
the dashboard. Each game is a list of board snapshots (one per turn)."""

from __future__ import annotations

import json

import numpy as np
import snek
import torch

from .net import AZNet
from .search import greedy_actions, mcts_search


@torch.no_grad()
def record_games(
    net: AZNet,
    device: torch.device,
    board: int = 11,
    n_games: int = 2,
    sims: int = 32,
    c_puct: float = 1.5,
    eval_batch_size: int = 8192,
    max_turns: int = 0,
    opponent: str = "baseline",  # "baseline" (snake1=flood-fill) or "net" (self-play)
    seed: int = 0,
) -> list[dict]:
    """Play `n_games` to completion, snapshotting every turn.

    Snake 0 is always the current net (greedy over the search policy). Snake 1 is
    the flood-fill baseline (`opponent="baseline"`) or the net again
    (`opponent="net"`, i.e. self-play). Returns a list of game dicts:
    `{opponent, winner, num_turns, frames: [snapshot, ...]}`.

    Uses the same MCTS (both policy and value heads) as self-play, so recorded
    replays reflect the net's real playing strength. The old equilibrium
    `run_search` here used the value head only, discarding the policy head where
    most of an MCTS-trained net's strength lives -- which made recorded games
    (esp. vs the baseline) look far weaker than the agent actually plays.
    """
    batch = snek.GameBatch(board, board, 2, count=n_games, seed=seed)
    frames: list[list[dict]] = [[] for _ in range(n_games)]
    # Stop recording a game once we've captured its terminal frame, so finished
    # games aren't padded with frozen frames up to the longest game's length.
    recorded_terminal = [False] * n_games

    steps = 0
    while max_turns <= 0 or steps < max_turns:
        done = batch.done().astype(bool)
        for g in range(n_games):
            if not recorded_terminal[g]:
                frames[g].append(json.loads(batch.snapshot(g)))
                if done[g]:
                    recorded_terminal[g] = True  # this was the final frame
        if all(recorded_terminal):
            break
        policy, _ = mcts_search(batch, net, device, sims=sims, c_puct=c_puct,
                                eval_batch_size=eval_batch_size)
        agent = greedy_actions(policy)[:, 0]
        if opponent == "net":
            opp = greedy_actions(policy)[:, 1]
        else:
            opp = batch.baseline_actions()[:, 1]
        actions = np.stack([agent, opp], axis=1).astype(np.uint8)
        batch.step(actions)
        steps += 1

    winners = batch.winners()
    return [
        {
            "opponent": opponent,
            "winner": int(winners[g]),
            "num_turns": len(frames[g]),
            "frames": frames[g],
        }
        for g in range(n_games)
    ]
