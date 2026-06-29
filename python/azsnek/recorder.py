"""Record full self-play / evaluation games as replayable frame sequences for
the dashboard. Each game is a list of board snapshots (one per turn)."""

from __future__ import annotations

import json

import numpy as np
import snek
import torch

from .net import AZNet
from .search import greedy_actions, mcts_search


def _snapshot_with_search(batch: snek.GameBatch, game_index: int, policy, value) -> dict:
    frame = json.loads(batch.snapshot(game_index))
    for i, snake in enumerate(frame.get("snakes", [])):
        snake["policy"] = [float(x) for x in policy[game_index, i]]
        snake["value"] = float(value[game_index, i])
    return frame


def _annotate_action(frame: dict, snake_index: int, move: int, play_policy) -> None:
    snakes = frame.get("snakes") or []
    if snake_index >= len(snakes):
        return
    snakes[snake_index]["chosen_move"] = int(move)
    snakes[snake_index]["play_policy"] = [float(x) for x in play_policy]


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
        terminal_now = []
        for g in range(n_games):
            if not recorded_terminal[g] and done[g]:
                frames[g].append(json.loads(batch.snapshot(g)))
                recorded_terminal[g] = True
                terminal_now.append(g)
        if all(recorded_terminal):
            break
        policy, value = mcts_search(batch, net, device, sims=sims, c_puct=c_puct,
                                    eval_batch_size=eval_batch_size)
        agent = greedy_actions(policy)[:, 0]
        if opponent == "net":
            opp = greedy_actions(policy)[:, 1]
        else:
            opp = batch.baseline_actions()[:, 1]
        for g in range(n_games):
            if not recorded_terminal[g] and g not in terminal_now:
                frame = _snapshot_with_search(batch, g, policy, value)
                _annotate_action(frame, 0, agent[g], policy[g, 0])
                _annotate_action(frame, 1, opp[g], policy[g, 1])
                frames[g].append(frame)
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
