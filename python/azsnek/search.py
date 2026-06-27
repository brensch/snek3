"""Bridge between the Rust fixed-depth equilibrium search and the PyTorch net.

A single search step is three calls:
  1. `batch.prepare_search(depth)` -> leaf observations `[M, C, H, W]`
  2. chunked batched net forward passes -> per-leaf values `[M]`
  3. `batch.backup_search(values, tau, iters)` -> root policies `[count, N, 4]`

Only the value head is needed at the leaves; the equilibrium backup turns those
into a policy. This is the whole anti-"simulation starvation" trick: one network
forward pass per search, no per-node evaluation.
"""

from __future__ import annotations

import numpy as np
import torch

from .net import AZNet, autocast as net_autocast


@torch.no_grad()
def run_search(
    batch,
    net: AZNet,
    device: torch.device,
    depth: int = 2,
    tau: float = 6.0,
    iters: int = 200,
    eval_batch_size: int = 8192,
    return_root_values: bool = False,
    temp=None,
):
    """Run one equilibrium search over every game in `batch`.

    Returns root policies `[count, num_snakes, 4]`. If `return_root_values` is
    set, returns `(policies, root_values)` where `root_values` is `[count,
    num_snakes]` — the per-agent equilibrium value of the current state.

    `temp` conditions a temperature-aware (proxy) value net at the leaves; it may
    be a scalar (all agents) or a per-agent array of length `num_snakes`. Leaf
    observations are laid out `[eval_id, agent]` with agent innermost.
    """
    obs = batch.prepare_search(depth)  # [M, C, H, W] float32
    if obs.shape[0] == 0:
        # Every root already terminal; backup still needs a (length-0) value array.
        values = np.zeros((0,), dtype=np.float32)
        if return_root_values:
            return batch.backup_search_values(values, tau, iters)
        return batch.backup_search(values, tau, iters)

    net.eval()
    use_temp = getattr(net.cfg, "temperature_input", False) and temp is not None
    temp_all = None
    if use_temp:
        n = int(batch.num_snakes)
        temp_arr = np.asarray(temp, dtype=np.float32).reshape(-1)
        if temp_arr.size == 1:
            temp_all = np.full(obs.shape[0], float(temp_arr[0]), dtype=np.float32)
        else:  # per-agent, tiled over leaves ([eval, agent], agent innermost)
            temp_all = np.tile(temp_arr, obs.shape[0] // n)
    if eval_batch_size <= 0:
        eval_batch_size = obs.shape[0]
    values = np.empty((obs.shape[0],), dtype=np.float32)
    for start in range(0, obs.shape[0], eval_batch_size):
        end = min(start + eval_batch_size, obs.shape[0])
        obs_t = torch.from_numpy(obs[start:end]).to(device, non_blocking=True)
        temp_t = (
            torch.from_numpy(temp_all[start:end]).to(device, non_blocking=True)
            if use_temp else None
        )
        with net_autocast(device):
            _, value = net(obs_t, temp_t)  # value: [M] in [-1, 1]
        values[start:end] = value.detach().to("cpu", dtype=torch.float32).numpy()
        del obs_t, value
    if return_root_values:
        return batch.backup_search_values(values, tau, iters)
    return batch.backup_search(values, tau, iters)


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
