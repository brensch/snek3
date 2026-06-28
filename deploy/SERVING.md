# Serving the Battlesnake `/move` API (Albatross-faithful, pure Rust)

This is the production play path: a small Rust binary (`crates/snek-server`) that
runs the **most Albatross-faithful** serving procedure on CPU, with no PyTorch and
no Python in the request loop.

## What it does on every `/move`

1. **Models the opponent online (temperature MLE).** It accumulates, per opponent
   and per candidate temperature on a log grid (`geomspace(0.25, 20, 24)`, the same
   `TAU_GRID` as training), the log-likelihood of that opponent's actual moves under
   the proxy net's policy head. A weak/predictable opponent scores a low `tau`. Only
   the newest move is scored each turn (the LL is additive), so it stays cheap as the
   game lengthens. Opponents are tracked by snake **id**, so other snakes dying (which
   shifts board indices) doesn't corrupt the estimate.
2. **Best-responds (heterogeneous-temperature equilibrium search).** It runs the
   fixed-depth logit-equilibrium search with **our** snake pinned at a high rational
   temperature (`SNEK_RESPONSE_TAU`) and each opponent pinned at its estimated `tau`.
   Leaves are evaluated by the proxy net in one batched ONNX forward, conditioned on
   the opponents' rationality regime. The equilibrium that comes back is our
   exploitative best response; we play its argmax.

This mirrors the earlier Albatross-style training design — opponent tau
estimation plus hetero-tau response policy — reimplemented over
`snek-infer`/ONNX. The **proxy** net is what serves (it does both the MLE and the
leaf eval); the *response* net is a not-yet-used distillation, so faithful serving
needs only the proxy + the search.

> First move(s) of a game have no opponent history yet, so each opponent starts at the
> grid's geometric-mean `tau` (~2.24) and the estimate sharpens as the game proceeds.

## Quick start (local)

```sh
make export-model          # runs/albatross-resp0/state.pt -> model.onnx (proxy net)
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

The image is a Rust binary + a CPU `onnxruntime` + `model.onnx` — no torch, no CUDA.
Point your Battlesnake at the box's `:8000`.

## Tuning (env vars)

| var | default | effect |
|-----|---------|--------|
| `SNEK_MODEL` | `model.onnx` | proxy ONNX path |
| `SNEK_PORT` | `8000` | listen port |
| `SNEK_THREADS` | `2` | request worker threads (inference is mutexed) |
| `SNEK_DEPTH` | `2` | search plies. Leaf count ≈ (legal^N)^depth; depth 3 is the practical ceiling |
| `SNEK_ITERS` | `120` | SFP iterations per node (equilibrium quality) |
| `SNEK_RESPONSE_TAU` | `12.0` | our rationality (higher = sharper best response) |
| `SNEK_DRAW_VALUE` | `-0.9` | leaf value of a draw (negative discourages mutual-suicide draws) |
| `SNEK_EVAL_CHUNK` | `4096` | max obs rows per ONNX forward |

Latency budget is ~500ms/move. If a CPU box is tight, lower `SNEK_DEPTH` (biggest lever)
or `SNEK_ITERS`; if you have headroom, raise `SNEK_DEPTH` to 3 for stronger tactics.

## Re-exporting after more training

The serving model is a snapshot. After the live run improves, re-run `make export-model`
(and rebuild the image) to ship the newer proxy. Architecture is read from the
checkpoint's `net_cfg`, so it always matches the weights.
