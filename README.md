# snek3 — AlphaZero-style Battlesnake bot

A Battlesnake agent built on AlphaZero ideas, adapted for **simultaneous moves**
using the *search* machinery from [Albatross](https://arxiv.org/abs/2402.03136):
fixed-depth, full-width search with a per-node **Logit Equilibrium** solve
(Stochastic Fictitious Play) instead of deep MCTS rollouts. This cures the
"simulation starvation" of simultaneous-move MCTS — the whole tree's leaves are
evaluated in a single batched neural-net forward pass per search.

We **assume (near-)perfect play**: the equilibrium solver runs at a high fixed
temperature (≈Nash). No bounded-rationality / opponent-temperature modeling, so
there is a single network and no two-stage training.

## Layout

| Path | What |
| --- | --- |
| `crates/snek-core` | Rules engine (faithful port of the official standard ruleset), observation encoding, standard board setup. Generic over snake count. |
| `crates/snek-search` | Fixed-depth equilibrium search + SFP solver *(Phase 4)*. |
| `crates/snek-py` | PyO3 bindings → the `snek` Python module (built with maturin). |
| `python/azsnek` | Network, self-play, training *(Phase 5)*. |
| `python/server` | FastAPI Battlesnake endpoint *(Phase 6)*. |
| `python/tests` | Binding tests. |

## Status

All phases implemented and tested (13 Python tests + 20 Rust tests green).

- [x] Phase 0 — workspace + maturin scaffolding
- [x] Phase 1 — Rust rules engine (12 fidelity tests, ~sub-µs/step)
- [x] Phase 2 — PyO3 bindings + observation encoding
- [x] Phase 3 — neural network (PyTorch ResNet, policy + value)
- [x] Phase 4 — fixed-depth equilibrium search + batched eval (Logit Equilibrium / SFP)
- [x] Phase 5 — self-play training loop (learns to beat the flood-fill baseline)
- [x] Phase 6 — Battlesnake FastAPI server
- [x] Phase 7 — FFA (N=4) runs through the same N-generic search

A 14-generation smoke run (~2 min on an RTX 5080, tiny net) took the agent from
losing every duel to the flood-fill baseline to losing none of 120
(0 → 0.16 → 0.33 → 0.43 → 0.50 win-rate, draws counted as half). Decisive wins
need a bigger net / deeper search / longer training / a win incentive.

## Train & serve

`maturin develop` installs `snek`, `azsnek`, and `server` into the venv as real
packages, so no `PYTHONPATH` juggling is needed.

```bash
# Train (writes checkpoints/best.pt on each eval improvement)
python -m azsnek.train --generations 50 --samples 20000 \
    --depth 2 --filters 64 --blocks 6 --eval-every 5

# Serve (filters/blocks must match the checkpoint)
SNEK_CKPT=checkpoints/best.pt SNEK_FILTERS=64 SNEK_BLOCKS=6 \
    uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## Build & test

```bash
# Rust core
cargo test -p snek-core
cargo bench -p snek-core         # step throughput

# Python boundary (in a venv)
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
maturin develop --release        # builds the `snek` module
python -m pytest python/tests -q
```

## Engine fidelity notes

The `step` pipeline matches `BattlesnakeOfficial/rules` ordering:
move → reduce health → hazard damage → **feed (grow) → eliminate**. Consequences
locked in by tests: shared food in a head-to-head keeps the tie (both grow), a
tail is stationary for one turn after eating, and a fully-vacating tail is safe
to enter.
