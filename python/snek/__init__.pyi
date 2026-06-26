"""Type stubs for the `snek` Rust extension module (built by maturin).

Hand-maintained; keep in sync with `crates/snek-py/src/lib.rs`. Without this,
static analyzers (Pylint/Pyright) cannot see members of the compiled module and
report false `no-member` (E1101) errors.
"""

from __future__ import annotations

import numpy as np

CHANNELS: int

class GameBatch:
    """A batch of independent Battlesnake games advanced in lockstep."""

    def __init__(
        self,
        width: int,
        height: int,
        num_snakes: int,
        count: int,
        seed: int = ...,
    ) -> None: ...
    @staticmethod
    def from_request(body: str) -> tuple[GameBatch, int]:
        """Build a single-game batch from a `/move` request; returns (batch, me)."""
        ...

    @property
    def count(self) -> int: ...
    @property
    def num_snakes(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def channels(self) -> int: ...
    def step(self, actions: np.ndarray) -> None:
        """Advance every non-terminal game. `actions`: uint8 [count, num_snakes]."""
        ...

    def encode(self) -> np.ndarray:
        """Egocentric observations: float32 [count, num_snakes, channels, H, W]."""
        ...

    def baseline_actions(self) -> np.ndarray:
        """Flood-fill baseline action per snake: uint8 [count, num_snakes]."""
        ...

    def alive(self) -> np.ndarray:
        """Per-snake alive mask: uint8 [count, num_snakes]."""
        ...

    def legal_moves(self) -> np.ndarray:
        """Non-reversal move mask: uint8 [count, num_snakes, 4]."""
        ...

    def done(self) -> np.ndarray:
        """Per-game terminal flag: uint8 [count]."""
        ...

    def winners(self) -> np.ndarray:
        """Per-game winner (-1 ongoing/draw else snake index): int8 [count]."""
        ...

    def prepare_search(self, depth: int) -> np.ndarray:
        """Search phase 1: leaf observations float32 [num_evals, channels, H, W]."""
        ...

    def backup_search(
        self, values: np.ndarray, tau: float = ..., iters: int = ...
    ) -> np.ndarray:
        """Search phase 2: root equilibrium policies float32 [count, num_snakes, 4]."""
        ...

    def reset_done(self) -> int:
        """Reset finished games to a fresh start; returns how many were reset."""
        ...

def encode_move_request(body: str) -> tuple[np.ndarray, int, np.ndarray]:
    """Parse a `/move` request -> (obs [channels,H,W], me_index, legal_mask [4])."""
    ...
