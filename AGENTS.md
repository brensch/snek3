# Agent Notes

## Repo Shape

- Rust rules and search live under `crates/snek-core` and `crates/snek-search`.
- The Rust trainer/API is `crates/snek-train`; it is a standalone Cargo project because it links libtorch through `tch`.
- The policy/value net is `crates/snek-tch`.
- The Battlesnake `/move` API is `crates/snek-server` and uses the same `snek-tch` checkpoint format as the trainer.
- The dashboard is the standalone Vite React TypeScript app in `frontend/`.
- Previous Python code is archived under `archived/` for reference only.
- Training outputs are runtime data under `runs/` and are ignored by git. Logs are under `logs/` and ignored.

## Common Commands

- `make test`: run top-level Rust tests.
- `make train START=1 RUN_ID=<id>`: build and run the Rust trainer/API.
- `make frontend`: run the Vite frontend, proxying `/api` to the trainer.
- `make frontend-build`: build the frontend to `frontend/dist`.
- `make api MODEL=runs/<run-id>/net.safetensors`: run the Battlesnake `/move` server.

## Libtorch / GPU Notes

`crates/snek-train`, `crates/snek-tch`, and `crates/snek-server` use `tch`, so
build/run with libtorch available. The current dev shortcut reuses the local
PyTorch libtorch:

```sh
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

The Makefile applies those variables for trainer and serving targets. A
standalone libtorch install can be used with `LIBTORCH=/path/to/libtorch`.

## Training State

Each Rust trainer run writes to `runs/<run-id>/`:

- `config.json`: single source of truth for knobs.
- `trainer_state.json`: generation, RNG seed/state metadata, best win rate, samples seen.
- `net.safetensors`: current network weights.
- `buffer/`: retained replay shards, restored on resume.
- `metrics.jsonl`: per-generation summaries.

## Current Training Interpretation

- `policy_loss` is cross-entropy to search policy targets. It cannot go below the entropy of those targets.
- `value_loss` is MSE from predicted value to final game outcome.
- `target_entropy` measures how spread out the search target policy is.
- Very high target entropy means targets are too soft/random. Very low entropy with a weak value net can mean overconfident bad targets.

## Dashboard Notes

- Frontend files should stay TypeScript, small, and single concern.
- Rebuild with `make frontend-build` after React/CSS changes.
- Realtime stats stream from `/api/stream/stats`.
