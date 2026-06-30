# Serving the Battlesnake `/move` API (pure Rust)

This is the production play path: a small Rust binary (`crates/snek-server`) that
runs the **same MCTS as self-play** on CPU, with no PyTorch and no Python in the
request loop — so what we serve matches what we trained.

## What it does on every `/move`

Run **decoupled-PUCT MCTS** over the policy+value net (one ONNX): the policy head
supplies per-snake priors, the value head evaluates leaves (no rollouts), and the
root visit counts decide the move (most-visited, death-masked). It is stateless
per move — board + our index only, no opponent modelling, no temperature.

Efficiency shortcuts baked into the search:

- **Forced move:** if our snake has a single legal move, return it immediately —
  zero sims, zero net forwards.
- **Leaf batching (virtual loss):** collect up to `SNEK_LEAVES_PER_SIM` distinct
  leaves per round and evaluate them in one forward, amortizing the ONNX-runtime
  call overhead on CPU.
- **Deadline-bound + early stop:** search until the request deadline (timeout −
  margin) or until the leading root move can no longer be overtaken in the
  remaining budget, then play it. `SNEK_MAX_SIMS` is only a safety cap.

## Quick start (local)

```sh
make export-model          # checkpoints/latest.pt -> model.onnx (policy+value net)
make api                   # builds + runs on :8000 using the venv's onnxruntime
# in another shell:
curl localhost:8000/
curl -X POST localhost:8000/move -d '{"game":{"id":"g"},"turn":0,"board":{"width":11,"height":11,"food":[],"snakes":[{"id":"me","health":99,"body":[{"x":1,"y":1},{"x":1,"y":2}]},{"id":"o","health":99,"body":[{"x":9,"y":9},{"x":9,"y":8}]}]},"you":{"id":"me","health":99,"body":[{"x":1,"y":1},{"x":1,"y":2}]}}'
```

Export a different checkpoint with `make export-model CHECKPOINT=runs/<id>/state.pt`.

## Deploy (CPU box, Docker)

```sh
make export-model          # produce ./model.onnx
make api-docker            # -> image snek-api  (CPU onnxruntime baked in)
docker run -p 8000:8000 snek-api
```

CI also builds and pushes `ghcr.io/brensch/snek3-api:latest` and
`ghcr.io/brensch/snek3-api:cpu` via `.github/workflows/api-image.yml`.

The image is a Rust binary + a CPU `onnxruntime` + `model.onnx` — no torch, no
CUDA in the runtime image. Point your Battlesnake at the box's `:8000`.

## Tuning (env vars)

| var | default | effect |
|-----|---------|--------|
| `SNEK_MODEL` | `model.onnx` | proxy ONNX path |
| `SNEK_PORT` | `8000` | listen port |
| `SNEK_THREADS` | `2` | request worker threads (inference is mutexed) |
| `SNEK_MAX_SIMS` | `100000` | safety cap on sims/move (serving is deadline-bound first) |
| `SNEK_LEAVES_PER_SIM` | `8` | leaves batched per selection round (virtual loss); 1 disables batching |
| `SNEK_VIRTUAL_LOSS` | `1.0` | virtual-loss magnitude steering batched descents apart |
| `SNEK_C_PUCT` | `1.5` | PUCT exploration constant (match training) |
| `SNEK_TIMEOUT_MS` | `500` | fallback per-move deadline when the request JSON omits `game.timeout` |
| `SNEK_DEADLINE_MARGIN_MS` | `150` | response margin reserved from the timeout |
| `SNEK_DRAW_VALUE` | `-0.25` | leaf value of a draw (match training) |
| `SNEK_EVAL_CHUNK` | `4096` | max obs rows per ONNX forward |

Latency budget is the request `game.timeout` (default ~500ms/move). On a tight CPU
box, raise `SNEK_LEAVES_PER_SIM` for more sims per forward; the search auto-stops
early once the leading move can't be overtaken.

## Game viewer

The server ships an embedded web viewer for the games it records (the compressed
`<game_id>.json.zst` logs under `SNEK_MOVE_LOG_DIR`). It is built from
`crates/snek-server/viewer/` and bundled into the binary via `rust-embed`
(`make api-build` and the Docker image build the frontend first; bare
`cargo build` uses a tracked placeholder).

- **UI:** `GET /app/` — pick a game, scrub/auto-play it, and read the per-turn
  search: value head, Up/Down/Left/Right policy + priors + visit counts + Q, the
  played move, sims/forward/eval/depth. Share a deep link with
  `/app/?game=<id>&turn=<n>`.
- **API** (namespaced under `/viewer/*`, separate from the Battlesnake routes):
  - `GET /viewer/games` — list recorded game ids
  - `GET /viewer/games/{id}` — the decompressed game JSON
  - `GET /viewer/games/{id}/tree?turn=N[&sims=M]` — replay turn `N` and return
    the full exploration tree (per-node PUCT decomposition Q + c·P·√ΣN/(1+N)).

Replay is a faithful reproduction: serving search is deterministic, so re-running
with the turn's recorded `sims_completed` rebuilds the in-game tree node-for-node
— **provided the loaded model matches the one that recorded the game**. Each game
stores a SHA-256 of the model; the tree response reports `model_match`, and the UI
warns when a replay used different weights (e.g. the model was re-exported since).
For an in-process replay (the same server that recorded the game) this always
matches. `SNEK_VIEWER_DIR` overrides the embedded assets with an on-disk dir for
iterating on the frontend without rebuilding the server.

## Re-exporting after more training

The serving model is a snapshot. After the live run improves, re-run `make export-model`
(and rebuild the image) to ship the newer proxy. Architecture is read from the
checkpoint's `net_cfg`, so it always matches the weights.
