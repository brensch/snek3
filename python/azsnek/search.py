"""Bridge between the Rust fixed-depth equilibrium search and the PyTorch net.

A single search step is three calls:
  1. `batch.prepare_search(depth)` -> leaf observations `[M, C, H, W]`
  2. one batched net forward pass -> per-leaf values `[M]`
  3. `batch.backup_search(values, tau, iters)` -> root policies `[count, N, 4]`

Only the value head is needed at the leaves; the equilibrium backup turns those
into a policy. This is the whole anti-"simulation starvation" trick: one network
forward pass per search, no per-node evaluation.
"""

from __future__ import annotations

import numpy as np
import torch

from .net import AZNet


@torch.no_grad()
def run_search(
    batch,
    net: AZNet,
    device: torch.device,
    depth: int = 2,
    tau: float = 6.0,
    iters: int = 200,
) -> np.ndarray:
    """Run one equilibrium search over every game in `batch`.

    Returns root policies as a float32 array of shape `[count, num_snakes, 4]`.
    """
    obs = batch.prepare_search(depth)  # [M, C, H, W] float32
    if obs.shape[0] == 0:
        # Every root already terminal; backup still needs a (length-0) value array.
        values = np.zeros((0,), dtype=np.float32)
        return batch.backup_search(values, tau, iters)

    net.eval()
    obs_t = torch.from_numpy(obs).to(device, non_blocking=True)
    _, value = net(obs_t)  # value: [M] in [-1, 1]
    values = value.detach().to("cpu", dtype=torch.float32).numpy()
    return batch.backup_search(values, tau, iters)


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
