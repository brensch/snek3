"""Bridge between the Rust batched MCTS forest and the PyTorch net.

`mcts_search` drives a decoupled-PUCT tree (`batch.mcts_*`): each simulation
selects leaves across all games, the net evaluates them in one batched forward
(both heads — policy for priors, value for the leaf estimate), and the visit
counts become the improved policy target. `sample_actions` / `greedy_actions`
turn a `[count, num_snakes, 4]` policy into per-snake moves.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .net import AZNet, autocast as net_autocast

# Lightweight per-call timing so the dashboard can show throughput / GPU-vs-CPU
# split. fwd_s = time in the net forward (GPU), search_s = time in the Rust
# tree-build + equilibrium backup (CPU), infer = leaf evaluations.
_PERF = {"fwd_s": 0.0, "search_s": 0.0, "infer": 0}


def perf_reset():
    _PERF.update(fwd_s=0.0, search_s=0.0, infer=0)


def perf_snapshot() -> dict:
    return dict(_PERF)


@torch.no_grad()
def mcts_search(
    batch,
    net: AZNet,
    device: torch.device,
    sims: int = 128,
    c_puct: float = 1.5,
    eval_batch_size: int = 8192,
    temp=None,
):
    """AlphaZero MCTS over every game in `batch` (decoupled-PUCT, simultaneous
    moves). Uses BOTH heads: policy = priors, value = leaf eval. Returns
    `(policies, root_values)` — visit-count policy targets `[count, num_snakes,
    4]` and mean root values `[count, num_snakes]`.

    `temp` optionally conditions a temperature-aware net at the leaves (scalar or
    per-agent length `num_snakes`).
    """
    net.eval()
    n = int(batch.num_snakes)
    use_temp = getattr(net.cfg, "temperature_input", False) and temp is not None
    temp_arr = None
    if use_temp:
        t = np.asarray(temp, dtype=np.float32).reshape(-1)
        temp_arr = t  # tiled per leaf below

    batch.mcts_new(c_puct)
    for _ in range(sims):
        pending, obs = batch.mcts_select()  # obs: [k, n, C, H, W]
        k = int(pending.shape[0])
        if k == 0:
            continue
        m = k * n
        flat = obs.reshape(m, *obs.shape[2:])
        leaf_temp = None
        if use_temp:
            leaf_temp = (
                np.full(m, float(temp_arr[0]), dtype=np.float32)
                if temp_arr.size == 1
                else np.tile(temp_arr, k)
            )
        pol_out = np.empty((m, 4), dtype=np.float32)
        val_out = np.empty((m,), dtype=np.float32)
        ebs = eval_batch_size if eval_batch_size > 0 else m
        for s in range(0, m, ebs):
            e = min(s + ebs, m)
            obs_t = torch.from_numpy(flat[s:e]).to(device, non_blocking=True)
            temp_t = (
                torch.from_numpy(leaf_temp[s:e]).to(device, non_blocking=True)
                if use_temp else None
            )
            with net_autocast(device):
                logits, value = net(obs_t, temp_t)
                probs = torch.softmax(logits.float(), dim=1)
            pol_out[s:e] = probs.cpu().numpy()
            val_out[s:e] = value.detach().float().cpu().numpy()
        batch.mcts_expand_backup(pending, pol_out.reshape(-1), val_out.reshape(-1))
    return batch.mcts_root_targets()


def sample_actions(policy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one action per snake from a `[count, N, 4]` policy (vectorized).

    Rows with no probability mass (terminal games / eliminated snakes) fall back
    to action 0; those moves are ignored by the engine anyway.
    """
    count, n, _ = policy.shape
    flat = policy.reshape(count * n, 4).astype(np.float64)
    sums = flat.sum(axis=1)
    safe = sums > 1e-8
    flat[safe] /= sums[safe, None]
    flat[~safe] = [1.0, 0.0, 0.0, 0.0]
    cdf = flat.cumsum(axis=1)
    u = rng.random((flat.shape[0], 1))
    # Index of the first bucket whose cumulative mass reaches u.
    actions = (u > cdf).sum(axis=1).clip(0, 3).astype(np.uint8)
    return actions.reshape(count, n)


def greedy_actions(policy: np.ndarray) -> np.ndarray:
    """Argmax action per snake from a `[count, N, 4]` policy."""
    return policy.argmax(axis=2).astype(np.uint8)
