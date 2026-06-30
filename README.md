# snek3

AlphaZero-style Battlesnake training and serving in Rust.

## Layout

| Path | What |
| --- | --- |
| `crates/snek-core` | Rules engine, standard board setup, observation encoding. |
| `crates/snek-search` | Simultaneous-move decoupled-PUCT MCTS. |
| `crates/snek-tch` | Libtorch/tch policy+value network used by the trainer. |
| `crates/snek-train` | Rust training binary plus realtime API for the frontend. |
| `crates/snek-server` | Battlesnake `/move` API server using the existing ONNX path. |
| `crates/snek-infer` | ONNX Runtime inference wrapper used by `snek-server`. |
| `frontend` | Standalone Vite/React dashboard. |
| `archived` | Previous Python trainer/dashboard/bindings kept for reference during the port. |

## Trainer

The trainer is a standalone crate so ordinary workspace tests do not need to link
libtorch. It owns self-play, training, checkpointing, and the dashboard API.

```bash
make train START=1 RUN_ID=dev
```

The API listens on `127.0.0.1:8050` by default.

Key routes:

| Route | Purpose |
| --- | --- |
| `GET /api/stream/stats` | protobuf `StatsFrame` over SSE, base64 encoded |
| `GET /api/stream/games` | protobuf `GamesSnapshot` over SSE, base64 encoded |
| `GET /api/state` | current run state |
| `GET/POST /api/config` | training knobs |
| `POST /api/control/start` | start or resume a run |
| `POST /api/control/stop` | graceful stop and checkpoint |
| `GET /api/runs` | list run directories |

Run state is written under `runs/<run-id>/`:

```text
config.json
trainer_state.json
net.safetensors
buffer/gen_*.json
metrics.jsonl
```

The replay buffer is restored from `buffer/` on resume.

## Frontend

```bash
make frontend
```

Vite proxies `/api` to the trainer on port `8050`. Production builds are written
to `frontend/dist`:

```bash
make frontend-build
```

## Libtorch

The trainer uses `tch`. For the current development setup, reuse the installed
PyTorch libtorch:

```bash
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

The Makefile applies those variables for trainer build/run targets. A fully
standalone libtorch install can be used by setting `LIBTORCH` instead.

## Commands

```text
make test             # top-level Rust tests
make train            # build and run snek-train
make frontend         # Vite dev server
make frontend-build   # static frontend build
make api              # existing Battlesnake /move server
make fmt
make lint
```
