# snek3 — AlphaZero-style Battlesnake bot

A Battlesnake agent built on AlphaZero, adapted for **simultaneous moves**:
**decoupled-PUCT MCTS** over a single policy+value network (no rollouts — the
value head evaluates leaves), with per-snake decoupled action selection so it
scales to N-player free-for-all. The same search runs in self-play training and
in the pure-Rust `/move` server, so what we serve is what we trained.

We **assume (near-)perfect play**: a single network, no bounded-rationality /
opponent-temperature modeling, no two-stage training.

## Layout

| Path | What |
| --- | --- |
| `crates/snek-core` | Rules engine (faithful port of the official standard ruleset), observation encoding, standard board setup. Generic over snake count. |
| `crates/snek-search` | Simultaneous-move MCTS (decoupled-PUCT) over the policy+value net *(Phase 4)*. |
| `crates/snek-py` | PyO3 bindings → the `snek` Python module (built with maturin). |
| `crates/snek-server` | Pure-Rust Battlesnake `/move` API — the same MCTS as self-play *(Phase 6)*. |
| `python/azsnek` | Network, self-play, training *(Phase 5)*. |
| `python/tests` | Binding tests. |

## Status

All phases implemented and tested (13 Python tests + 20 Rust tests green).

- [x] Phase 0 — workspace + maturin scaffolding
- [x] Phase 1 — Rust rules engine (12 fidelity tests, ~sub-µs/step)
- [x] Phase 2 — PyO3 bindings + observation encoding
- [x] Phase 3 — neural network (PyTorch ResNet, policy + value)
- [x] Phase 4 — simultaneous-move MCTS (decoupled-PUCT) + batched leaf eval
- [x] Phase 5 — self-play training loop (learns to beat the flood-fill baseline)
- [x] Phase 6 — pure-Rust Battlesnake `/move` server (same MCTS as self-play)
- [x] Phase 7 — FFA (N=4) runs through the same N-generic search

A 14-generation smoke run (~2 min on an RTX 5080, tiny net) took the agent from
losing every duel to the flood-fill baseline to losing none of 120
(0 → 0.16 → 0.33 → 0.43 → 0.50 win-rate, draws counted as half). Decisive wins
need a bigger net / deeper search / longer training / a win incentive.

## Train & serve

`maturin develop` installs `snek` and `azsnek` into the venv as real packages, so
no `PYTHONPATH` juggling is needed.

```bash
# Train (writes runs/<run_id>/ckpt/best.pt on each eval improvement)
python -m azsnek.train --generations 50 --samples 50000 \
    --sims 500 --filters 64 --blocks 6

# Serve the /move API (pure-Rust, same MCTS as self-play):
make export-model && make api      # or: make api-run
```

## Real-server arena

The Rust API can also orchestrate local Battlesnake games in-process while
serving the same `/app` replay UI. Every arena move is made by constructing a
real Battlesnake `/move` request with `game.timeout = 100`, passing it through
the same server move path as HTTP serving, then ending the game through the same
recorder. Logs are written as the server's normal `.json.zst` game files and are
available at `http://127.0.0.1:8000/app`.

```bash
# Export old generation snapshots if you want old-vs-new matchups.
PYTHONPATH=python .venv/bin/python scripts/export_model.py \
  runs/myrun/models/gen_0005.pt checkpoints/gen_0005.onnx
PYTHONPATH=python .venv/bin/python scripts/export_model.py \
  runs/myrun/models/gen_0050.pt checkpoints/gen_0050.onnx

# Play 20 two-player games at the real 100ms move timeout.
make arena ARENA_GAMES=20 \
  ARENA_MODELS="checkpoints/gen_0005.onnx,checkpoints/gen_0050.onnx" \
  ARENA_NAMES="gen5,gen50"
```

Each participant gets its own server/recorder instance, so the saved files are
participant-perspective logs like `arena-1-0000-s0.json.zst` and
`arena-1-0000-s1.json.zst` in `logs/api_moves/`.

## Live dashboard

Training streams progress to `runs/<run_id>/` as it goes: `metrics.jsonl` (per
generation), `status.json` (latest), and `games/gen_XXXX.json` (recorded replays
— self-play *and* vs-baseline). The dashboard reads those files; nothing is
precomputed.

```bash
# In one terminal: train (add --run-id to name the run)
python -m azsnek.train --generations 50 --samples 50000 --eval-every 1 --run-id myrun

# In another: the dashboard, then open http://127.0.0.1:8050
SNEK_RUNS_DIR=runs uvicorn dashboard.app:app --port 8050
```

It shows the win-rate / loss curves updating live and an auto-streaming game
viewer that plays the newest recorded games back-to-back (scrub, speed control,
"stream newest" toggle). The UI is a React app (`python/dashboard/ui`, built with
Vite); the built bundle is committed to `python/dashboard/static`, so the server
needs no Node at runtime. Rebuild after editing the UI with `make ui`.

## Make targets

Everything is wrapped in a `Makefile` (`make` or `make help` lists it):

```text
make venv        # create .venv + install deps (incl. PyTorch; TORCH_INDEX overridable)
make build       # compile the Rust extension into the venv
make test        # Rust + Python tests
make train       # train; writes runs/<id>/  (override GENERATIONS, SAMPLES, RUN_ID, ARGS...)
make dashboard   # serve the live dashboard (PORT, default 8050)
make api         # build + run the pure-Rust /move server (after make export-model)
make bench lint fmt clean clean-all
```

Typical first run: `make venv && make build && make test`, then
`make train RUN_ID=myrun` in one terminal and `make dashboard` in another.

### Resuming a run (automatic)

Each generation, the trainer atomically writes full training state (model +
optimizer + generation counter + best win-rate + RNG) to `runs/<run-id>/state.pt`.
It also saves per-generation model snapshots to
`runs/<run-id>/models/gen_XXXX.pt` for comparing against older versions of the
same snake.
**Re-running with the same `--run-id` resumes automatically** — it restores the
net architecture from the saved state, continues the generation numbering, and
appends to the same `metrics.jsonl`/`games/`:

```bash
make train RUN_ID=myrun GENERATIONS=100   # continues where it left off
make train RUN_ID=myrun FRESH=1           # ignore saved state, restart from scratch
```

A brand-new `--run-id` (or the default timestamp id) starts fresh. The
weights-only `ckpt/latest.pt` and `ckpt/best.pt` inside the run directory are for
serving; `state.pt` is the one used for resuming. Set `--ckpt-dir` only when you
intentionally want to write serving weights somewhere else.

## Engine fidelity notes

The `step` pipeline matches `BattlesnakeOfficial/rules` ordering:
move → reduce health → hazard damage → **feed (grow) → eliminate**. Consequences
locked in by tests: shared food in a head-to-head keeps the tie (both grow), a
tail is stationary for one turn after eating, and a fully-vacating tail is safe
to enter.
