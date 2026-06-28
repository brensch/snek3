"""Observation channel schema — the single source of truth for the board
encoding, shared by the Rust encoder (`crates/snek-core/src/encode.rs`) and the
PyTorch net. Keep the two in lockstep: if you change `PLANES` here, mirror it in
`encode.rs` and bump `SCHEMA_VERSION`.

Design goals (multiplayer-first):
  * **Permutation-invariant over opponents** — opponents are unioned for spatial
    occupancy, and per-opponent scalars (health, relative length) are written at
    each opponent's *head* cell. So the channel count is FIXED for any snake
    count N (1v1, 3-player, 4-player FFA all use the same tensor), and the net
    never depends on an arbitrary opponent ordering.
  * **Absolute board coordinates** (not head-centered): walls/food/hazards are
    absolute, and the net locates itself via the `my_head` plane + global pooling.
    This is cheaper than a 2*side-1 egocentric canvas and board-size agnostic.
  * Carries the things needed for non-degenerate play: **health** (starvation),
    **food**, **hazards**, **relative length** (head-to-head), and **tail
    countdown** (a body cell about to vacate is safe to enter).

Each plane is one [H, W] float map, encoded from snake `me`'s perspective.
"""
from __future__ import annotations

SCHEMA_VERSION = 1

# (name, description). Order defines the channel index.
PLANES: list[tuple[str, str]] = [
    ("my_head",            "1 at my head cell"),
    ("my_body",            "1 at my body segments (excl. head)"),
    ("my_tail_countdown",  "my body cells: (1 - turns_until_vacated/len), ~1 near the tail"),
    ("my_health",          "my health/100, broadcast to every cell"),
    ("my_length",          "my length/area, broadcast to every cell"),
    ("opp_heads",          "union of all live opponents' head cells"),
    ("opp_body",           "union of all live opponents' body segments (excl. heads)"),
    ("opp_tail_countdown", "union of opponents' body cells: (1 - turns_until_vacated/len)"),
    ("opp_len_vs_me",      "per-opponent (their_len - my_len)/area, written at their head cell"),
    ("opp_health",         "per-opponent health/100, written at their head cell"),
    ("opp_danger_heads",   "heads of opponents at least as long as me (would win/tie head-to-head)"),
    ("food",               "1 at food cells"),
    ("hazards",            "1 at hazard cells"),
    ("board_mask",         "1 over real board cells (for non-square / padded canvases)"),
]

CHANNELS: int = len(PLANES)  # = 14

# index-by-name for callers/tests
INDEX: dict[str, int] = {name: i for i, (name, _) in enumerate(PLANES)}
