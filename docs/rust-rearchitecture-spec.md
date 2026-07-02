# snek3 — Rust-only rearchitecture spec

Status: proposed (no implementation yet). Author: Claude. Date: 2026-06-30.

## 1. Goal & principles

Remove Python from the repo entirely and reimplement everything it did in Rust, as
a single training binary with a streaming API for a standalone frontend.

Principles:

- **One process, one net.** The tch (libtorch) net is used for *both* training and
  self-play inference, in-process. No ONNX export, no `onnxruntime`, no numpy
  handoff, no PyO3.
- **Keep what's proven, delete the scaffolding.** The MCTS engine and the
  GPU/CPU shard double-buffer pipeline rail the GPU (~53–57k rows/s in isolation)
  and are kept as-is. The bloat being removed is around it: 4-layer config, the
  1.6 GB continuous self-play state, dual MCTS paths, the `--serve` defaulting +
  auto-resume-latest surprises, the Python trainer/dashboard.
- **Validated foundation.** The net is already ported and verified
  (`crates/snek-tch`): identical architecture/param count (1,398,917), GroupNorm
  eps 1e-5, orthogonal(√2) init (manual, since tch's builtin corrupts conv
  shapes), and **71k rows/s @512 — faster than the current ort path**.
- **State storage is a first-class concern.** Resume must restore net + optimizer
  + generation + RNG + replay buffer deterministically.

Non-goals (for this effort): changing the learning algorithm; replacing the
deployed `/move` server (`crates/snek-server`, Rust+ort) — it stays, ort is not
Python; rewriting the MCTS math.

## 2. Current state (what exists)

| Component | Where | Disposition |
|---|---|---|
| Rules / board / encode | `crates/snek-core` | keep |
| MCTS forest (`MctsForest`) | `crates/snek-search` | keep |
| Self-play loop `generate_selfplay` | `crates/snek-py` (PyO3) | **port** to pure Rust (swap ort→tch), then archive crate |
| ort inference wrapper | `crates/snek-infer` | removed after `snek-server` moved to tch |
| tch net (`AZNet`, `init_orthogonal`) | `crates/snek-tch` | **keep — core of new trainer** |
| burn net + bench | `crates/snek-burn` | removed |
| `/move` API + arena | `crates/snek-server` | keep; now uses `snek-tch` |
| Python trainer / dashboard / tests | `python/` | **archive** |
| React/Vite dashboard UI | `python/dashboard/ui` | **move** to `/frontend`, repoint at Rust API |

## 3. Target layout

```
crates/
  snek-core/      (unchanged)
  snek-search/    (unchanged)
  snek-tch/       net + training primitives (grow this)
  snek-train/     NEW binary: self-play + train + API + orchestration
  snek-server/    deployed /move server, using snek-tch
frontend/         NEW: standalone Vite app (moved from python/dashboard/ui)
archived/
  python/         (everything from python/, for reference)
  pyproject.toml, export_model.py
docs/
  rust-rearchitecture-spec.md   (this file)
```

`snek-train` is a standalone crate (its own `[workspace]`, like `snek-tch`)
that path-depends on `snek-core`, `snek-search`, `snek-tch`, and links libtorch.

## 4. `snek-train` internals

### 4.1 Module layout

```
snek-train/src/
  main.rs        CLI + wiring (start API, spawn trainer task)
  config.rs      RunConfig (single source of truth) + load/save/merge
  net.rs         re-export snek-tch AZNet; forward helpers for inference & train
  selfplay.rs    pure-Rust port of generate_selfplay (tch inference)
  replay.rs      ReplayBuffer (ring of sample shards) + recency/balanced sampling
  train.rs       training step: loss, Adam, D4 augmentation, autocast
  trainer.rs     orchestration loop: gen cycle, phase machine, stop/start
  state.rs       checkpoint/resume (net, opt, gen, rng, buffer)
  api/
    mod.rs       axum router, app state
    sse.rs       stats + games SSE streams
    control.rs   start/stop/config endpoints
    proto.rs     prost-generated types (build.rs compiles proto/*.proto)
  metrics.rs     shared atomics/broadcast channels feeding the API
proto/
  snek.proto     protobuf schema (frontend + backend share this)
build.rs         prost-build compiles proto/snek.proto
```

### 4.2 Self-play (`selfplay.rs`)

Direct port of `crates/snek-py/src/lib.rs::generate_selfplay`, minus PyO3/numpy:

- Keep the **GPU worker thread + mpsc channel + N-shard double buffer** exactly
  (this is what rails the GPU). The worker owns the tch net on CUDA and runs
  `forward` on `[m, 14, 11, 11]` batches.
- Keep `MctsForest` select / write_pending_obs / expand_backup / freeze_forced_roots
  / root_targets unchanged (snek-search).
- Replace ort `Net::forward` with tch: host obs `Vec<f32>` → `Tensor::from_slice`
  → CUDA → `net.forward` under `tch::no_grad` + bf16 autocast → policy softmax +
  value back to `Vec<f32>`. (The GPU worker holds the net; training holds the same
  VarStore — see 4.5 on weight sharing/snapshotting.)
- Drop **continuous self-play state** (`selfplay_state.bin`, 1.6 GB): each
  generation plays fresh games to `samples_per_gen`. (Forest construction is
  cheap; this removes the v1/v2 bincode serializer entirely.)
- Keep length-balanced sampling (`balanced_sample_output`) and the optional
  in-loop game recorder (used by the games stream / replay).
- Emit live metrics every step into `metrics.rs` (see 4.7): inferences, gpu busy,
  completed games, per-game turn counts, batch fill.

Output per generation: `(obs [N,14,11,11], pol [N,4], z [N])` as Rust `Vec<f32>`
fed straight into the replay buffer — no serialization boundary.

### 4.3 Replay buffer (`replay.rs`)

Port of `python/azsnek/selfplay.py::ReplayBuffer`: a `VecDeque` of per-generation
`Samples { obs, pol, z }` arrays in host RAM, evicting oldest past `buffer_size`.
- `add(samples)`; `len()`; capacity eviction.
- Sampling: recency-biased index draw (`r**recency`) matching `train.py:336-342`.
- D4 symmetry augmentation on the sampled minibatch (port `symmetry.py`).
- **Persisted** as part of state (see 4.6) — restarts must not lose the recency
  window (today they cost ~20–30 gens of flat win-rate).

### 4.4 Training step (`train.rs`)

Port of `train.py::train_on_samples`:
- For `train_steps` iterations: sample minibatch (recency), D4-augment on CPU,
  `to(cuda, non_blocking)`, forward under bf16 autocast, losses:
  - policy: `-(target * log_softmax(logits)).sum(1).mean()` (soft cross-entropy)
  - value: `mse(value, z)`
  - `loss = policy_loss + value_weight * value_loss`
- Optimizer: tch `nn::Adam` (lr, betas, eps matching the Python optimizer).
- TF32 on (`tch::Cuda` cudnn benchmark; matmul/cudnn allow_tf32) to match
  `device_auto()`.
- Parity gate: on a fixed batch of real self-play data, Rust losses track the
  Python trainer's within tolerance over a few steps before we trust it.

### 4.5 Net ownership & weight sharing

Training mutates the VarStore; self-play needs current weights. Options (spec
picks **A**):

- **A. Single VarStore, snapshot per generation.** Train phase and play phase are
  sequential (as today). Before each play phase, the GPU worker uses the current
  net (same VarStore on CUDA). No copy needed because phases don't overlap. The
  worker thread holds an `Arc` to the net module; training happens between play
  phases on the same tensors. (If we later overlap phases, switch to B.)
- B. Double-buffered weights: training writes to VS_train; copy to VS_play
  (`VarStore::copy`) at generation boundary. Deferred unless we pipeline.

### 4.6 State storage & resume (`state.rs`)

A run directory `runs/<id>/` containing:

```
config.json         the single RunConfig (replaces meta.json + params.json)
net.safetensors     VarStore weights (tch save)
optim.ot            optimizer state (tch)
buffer/             replay shards (one file per retained generation)
trainer_state.json  { generation, rng_seed/state, best_winrate, samples_seen }
metrics.jsonl       append-only per-generation metrics (history for the UI)
games/              optional recorded games for replay
```

Resume = load config, net, optimizer, trainer_state, and rebuild the replay buffer
from `buffer/` shards. Atomic writes (temp + rename). No more `state.pt` blob, no
`selfplay_state.bin`.

### 4.7 Metrics plumbing (`metrics.rs`)

A shared `Metrics` struct: atomics for the cheap counters (inferences, completed
games, gpu busy us/idle us, current gen/phase) updated by self-play hot path with
`Relaxed` ordering (same as today's progress atomics). A `tokio::sync::broadcast`
channel carries structured `StatsFrame`s at a fixed cadence (e.g. 4 Hz) to SSE
subscribers. A separate, lower-cadence sampler builds `GamesSnapshot`s from the
live forest/board state when at least one games-stream subscriber is connected
(zero cost when nobody's watching).

Derived rates (inf/s, games/s) are computed from deltas in the sampler, not the
hot path.

### 4.8 Orchestration & lifecycle (`trainer.rs`)

Phase machine per generation: `PLAYING → TRAINING → CHECKPOINT → (repeat)`.
- `RunState { Idle, Playing, Training, Stopping, Stopped }` behind a watch channel.
- **Start/stop**: control endpoint sets a flag; self-play checks it between sims
  (like today's `check_cancelled`) and trainer checks between steps; on stop it
  finishes the current micro-step, checkpoints, and parks in `Stopped`.
- Knob changes (lr, sims, etc.) applied at the next generation boundary (mirrors
  current adaptive override behavior, minus the autotune subprocess).

## 5. API design (`api/`)

`axum` + `tokio`. CORS enabled (frontend is a separate origin/dev server). All
realtime data is protobuf; control is JSON for ergonomics.

### 5.1 Routes

| Method | Path | Body/Resp | Purpose |
|---|---|---|---|
| GET | `/api/stream/stats` | SSE of `StatsFrame` | high-freq training/perf stats |
| GET | `/api/stream/games` | SSE of `GamesSnapshot` | per-game state for viz (on-demand) |
| GET | `/api/state` | `RunState` (proto) | current snapshot for initial load |
| GET | `/api/config` | `RunConfig` (json) | current knobs |
| POST | `/api/config` | `RunConfig` (json) | update knobs (applied next gen) |
| POST | `/api/control/start` | `{run_id?, fresh?}` | start/resume a run |
| POST | `/api/control/stop` | — | graceful stop + checkpoint |
| GET | `/api/runs` | list | available runs to resume |
| GET | `/api/metrics/history` | `metrics.jsonl` | backfill charts on load |

### 5.2 Protobuf over SSE

SSE `data:` is UTF-8 text, so each protobuf frame is **base64-encoded** in the
event data, with the `event:` field naming the type. Overhead (~33%) is fine at
the stats cadence; `GamesSnapshot` uses a compact encoding (positions as varint
packed fields) to stay small even at 512 games. WebSocket (binary) is the
fallback if the games stream proves too heavy — schema is identical either way.

### 5.3 `proto/snek.proto` (sketch)

```proto
syntax = "proto3";
package snek;

message StatsFrame {
  uint64 t_unix_ms = 1;
  uint32 generation = 2;
  Phase  phase = 3;
  double inferences_per_sec = 4;
  double games_per_sec = 5;
  uint64 completed_games_total = 6;
  uint32 samples_collected = 7;
  uint32 samples_target = 8;
  double gpu_busy_pct = 9;
  uint32 batch_avg_rows = 10;
  // populated during TRAINING:
  double policy_loss = 11;
  double value_loss = 12;
  double target_entropy = 13;
}

enum Phase { IDLE=0; PLAYING=1; TRAINING=2; CHECKPOINT=3; STOPPING=4; STOPPED=5; }

message GameSnapshot {
  uint32 id = 1;
  uint32 turn = 2;            // steps so far
  uint32 board_w = 3;
  uint32 board_h = 4;
  repeated Snake snakes = 5;
  repeated uint32 food = 6;   // packed x,y
}
message Snake { bool alive = 1; uint32 health = 2; repeated uint32 body = 3; } // packed x,y
message GamesSnapshot { uint64 t_unix_ms = 1; repeated GameSnapshot games = 2; }

message GenerationSummary {        // appended to history, also pushed once per gen
  uint32 generation = 1;
  double policy_loss = 2; double value_loss = 3;
  double win_rate = 4; uint32 completed_games = 5; double seconds = 6;
}

message RunState { Phase phase = 1; uint32 generation = 2; string run_id = 3; bool running = 4; }
```

`RunConfig` stays JSON (it's a config object the UI edits), not proto.

## 6. Config (`config.rs`)

Single `RunConfig` struct = the union of today's `default_params.json` (27 keys)
that actually matter, serialized to `runs/<id>/config.json`. **No 4-layer
override.** Precedence: explicit CLI flag > config.json (on resume) > built-in
defaults. Fields (initial set, mirrors current knobs the UI shows):

```
board, num_snakes, count, sims, c_puct, gpu_batch_games,
samples_per_gen, exploration_prob, max_turns, draw_value,
skip_short_draw_turns, bootstrap_value,
trunk_channels, trunk_blocks, gpool_every,
train_steps, batch_size, lr, recency, buffer_size, value_weight,
search_threads, record_games, eval_every, eval_games
```

The frontend's existing `paramInfo.js` descriptions are reused for the knob UI.

## 7. Frontend (`/frontend`)

- Move `python/dashboard/ui/*` → `/frontend` unchanged, then:
  - Replace the in-process polling with **SSE subscriptions** (`/api/stream/stats`,
    `/api/stream/games`) and protobuf decode (`protobuf-es` or `ts-proto`,
    generated from the same `snek.proto`).
  - REST for config GET/POST and start/stop.
  - Keep the existing knobs/`paramInfo`; wire them to `POST /api/config`.
- Realtime views:
  - **Header stats**: inf/s, games/s, completed games, gen/phase, losses (from
    `StatsFrame`, ~4 Hz).
  - **Charts**: backfilled from `/api/metrics/history`, appended from
    `GenerationSummary`.
  - **Game grid**: render every live game from `GamesSnapshot` (board + snakes +
    per-game step count), refreshing at the games-stream cadence. Subscribing
    starts the snapshot sampler; leaving the view stops it (no backend cost when
    unwatched).
- Dev: `vite dev` proxies `/api` to the trainer (configurable base URL). Build is
  a static bundle deployable anywhere; the binary does **not** serve it.

## 8. Build & libtorch

- `snek-train` links libtorch via tch. Two supported modes:
  - **Dev (reuse venv torch):** `LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1`,
    runtime `LD_PRELOAD=libtorch_global_deps.so:libtorch_cuda.so`,
    `LD_LIBRARY_PATH` over `torch/lib` + `nvidia/*/lib` + WSL `libcuda`.
    (This is what the validated benches use; documented in AGENTS.md.)
  - **Standalone:** `tch` `download-libtorch` feature pinned to a supported
    libtorch+cuDNN; no Python in the loop at all. Preferred end state.
- `Makefile` rewritten: `make train` builds & runs `snek-train`; `make frontend`
  runs Vite; drop all maturin/azsnek targets. (The 15 python references go.)
- A small wrapper script exports the libtorch env so `cargo run -p snek-train`
  "just works" (mirrors how `_setup_ort_env()` does it for the current trainer).

## 9. Migration sequencing (maps to backlog tasks)

1. **Archive** `python/` → `archived/`; move UI → `/frontend`. (repo still builds
   the Rust side; training is offline until step 7.)
2. **Scaffold** `snek-train` (deps, config, net wire) — compiles.
3. **Training step** in tch + parity check against real data.
4. **Port self-play** (tch inference, drop continuous state) + metrics emission.
5. **Replay buffer** + recency/balanced sampling + D4 augmentation.
6. **Checkpoint/resume** + run-dir state layout.
7. **Orchestration loop** (phases, start/stop) — first end-to-end Rust training.
8. **API**: axum + SSE + proto + control endpoints.
9. **Frontend**: repoint to API, protobuf decode, realtime game grid.
10. **Cleanup**: rewrite Makefile, remove `snek-py`, prune `snek-infer` from the
    trainer path, update docs/AGENTS.md.

Each step is independently reviewable; training comes back online at step 7, the
UI at step 9.

## 10. Risks & open questions

- **Training parity.** Adam/loss/augmentation must reproduce learning behavior.
  Mitigation: step-3 parity gate on real data before trusting the loop. Bit-exact
  is impossible (RNG/kernels differ); we validate loss curves + win-rate, not
  weights.
- **tch vs torch 2.11.** Version check bypassed and works for inference; for
  training (autograd/optim) confirm no ABI surprises, else pin libtorch via
  `download-libtorch`.
- **libtorch deployment.** The trainer now needs libtorch present at runtime
  (`LD_PRELOAD` quirk). Fine for the training box; documented.
- **Games stream volume.** 512 games at a few Hz over base64 SSE — measure; fall
  back to WebSocket/binary if needed (schema unchanged).
- **`snek-server` divergence.** Superseded: serving now uses the same
  `snek-tch` checkpoint format as training.
- **Buffer persistence size.** Persisting the replay buffer (step 5/6) trades disk
  for resume quality; cap and shard like today's `buffer/`.

## 11. What is already done (re-usable now)

- `crates/snek-tch`: exact, validated `AZNet` + correct `init_orthogonal` + a
  throughput bench (71k rows/s @512). This is the net the trainer will use.
- GPU/libtorch env recipe proven and recorded in AGENTS.md.
- burn/cubecl rejected with data (~12× too slow); the experimental crate was
  removed once the tch path became canonical.
```
