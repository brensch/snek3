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

## Live dashboard

Training streams progress to `runs/<run_id>/` as it goes: `metrics.jsonl` (per
generation), `status.json` (latest), and `games/gen_XXXX.json` (recorded replays
— self-play *and* vs-baseline). The dashboard reads those files; nothing is
precomputed.

```bash
# In one terminal: train (add --run-id to name the run)
python -m azsnek.train --generations 50 --samples 20000 --eval-every 5 --run-id myrun

# In another: the dashboard, then open http://127.0.0.1:8050
SNEK_RUNS_DIR=runs uvicorn dashboard.app:app --port 8050
```

It shows the win-rate / loss curves updating live and a game viewer where you can
pick a generation + game and scrub/play through the recorded board states (snake
bodies, heads, food, health, who won). "Follow latest" auto-jumps to the newest
replays as training advances.

## Make targets

Everything is wrapped in a `Makefile` (`make` or `make help` lists it):

```text
make venv        # create .venv + install deps (incl. PyTorch; TORCH_INDEX overridable)
make build       # compile the Rust extension into the venv
make test        # Rust + Python tests
make train       # train; writes runs/<id>/  (override GENERATIONS, SAMPLES, RUN_ID, ARGS...)
make dashboard   # serve the live dashboard (PORT, default 8050)
make serve       # run the Battlesnake server (CKPT=..., matching FILTERS/BLOCKS)
make audit       # full end-to-end audit script
make bench lint fmt clean clean-all
```

Typical first run: `make venv && make build && make test`, then
`make train RUN_ID=myrun` in one terminal and `make dashboard` in another.

## Engine fidelity notes

The `step` pipeline matches `BattlesnakeOfficial/rules` ordering:
move → reduce health → hazard damage → **feed (grow) → eliminate**. Consequences
locked in by tests: shared food in a head-to-head keeps the tie (both grow), a
tail is stationary for one turn after eating, and a fully-vacating tail is safe
to enter.
