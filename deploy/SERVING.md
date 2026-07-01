# Serving the Battlesnake `/move` API

`crates/snek-server` serves Battlesnake moves with the same Rust MCTS and
`snek-tch` policy/value network used by training. It loads `net.safetensors`
checkpoints written by `snek-train`; there is no ONNX export or ONNX Runtime path.

## Quick Start

```sh
make api MODEL=runs/<run-id>/net.safetensors
curl localhost:8000/
```

The Makefile applies the same libtorch environment as `make train`.

## Runtime Config

| var | default | effect |
| --- | --- | --- |
| `SNEK_MODEL` | `net.safetensors` | tch checkpoint path |
| `SNEK_PORT` | `8000` | listen port |
| `SNEK_THREADS` | `2` | request worker threads; inference is mutexed |
| `SNEK_MAX_SIMS` | `100000` | safety cap on sims/move; serving is deadline-bound first |
| `SNEK_LEAVES_PER_SIM` | `8` | leaves batched per selection round with virtual loss |
| `SNEK_VIRTUAL_LOSS` | `1.0` | virtual-loss magnitude |
| `SNEK_C_PUCT` | `1.5` | PUCT exploration constant |
| `SNEK_TIMEOUT_MS` | `500` | fallback deadline when request JSON omits `game.timeout` |
| `SNEK_DEADLINE_MARGIN_MS` | `150` | response margin reserved from timeout |
| `SNEK_DRAW_VALUE` | `-0.25` | leaf value of a draw |
| `SNEK_EVAL_CHUNK` | `4096` | max obs rows per net forward |
| `SNEK_TRUNK_CHANNELS` | `96` | network width; must match checkpoint |
| `SNEK_TRUNK_BLOCKS` | `8` | network depth; must match checkpoint |
| `SNEK_GPOOL_EVERY` | `3` | global-pool cadence; must match checkpoint |
| `SNEK_CPU_ONLY` | unset | set to `1` to force CPU serving |

## Viewer

The server ships an embedded viewer for compressed game logs under
`SNEK_MOVE_LOG_DIR`.

- `GET /app/` serves the UI.
- `GET /viewer/games` lists recorded games.
- `GET /viewer/games/{id}` returns one decompressed game.
- `GET /viewer/games/{id}/tree?turn=N[&sims=M]` replays a turn and returns the
  search tree.

Each recorded game stores a SHA-256 of the model file. Tree replay reports
whether the current model matches the one used for the recorded game.
