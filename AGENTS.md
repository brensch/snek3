# Agent Notes

## Repo Shape

- Rust rules/search bindings live under `crates/`; build them into Python with `make build` (`maturin develop --release`).
- Python training code lives in `python/azsnek/`.
- The FastAPI Battlesnake server is in `python/server/`.
- The dashboard backend is in `python/dashboard/`; the React UI source is `python/dashboard/ui/`, with committed built assets in `python/dashboard/static/`.
- Training outputs are runtime data under `runs/` and are ignored by git. Logs are under `logs/` and ignored.

## Common Commands

- `make build`: compile the Rust Python extension.
- `make test-py`: build and run Python tests.
- `make test`: run Rust and Python tests.
- `make train`: foreground fixed-parameter training.
- `make overnight`: background fixed-parameter training via `scripts/overnight_train.sh`.
- `make adaptive`: foreground adaptive training; Ctrl-C stops it.
- `make dashboard`: serve the training dashboard on `PORT` default `8050`.

## Training State

- Each run writes to `runs/<run-id>/`.
- `state.pt` is the full resumable training state: network, optimizer, RNG, generation, and best win rate.
- Serving checkpoints are per-run in `runs/<run-id>/ckpt/latest.pt` and `best.pt`.
- `metrics.jsonl` is one JSON object per generation and feeds the dashboard.
- `meta.json` records run config and may be updated by adaptive tuning.
- The replay buffer is currently in memory only. Restarting `azsnek.train` resumes the net/optimizer but does not restore the replay buffer.

## Adaptive Training

Adaptive tuning is implemented in-process inside `python/azsnek/train.py` via `--adaptive`. This is important: the earlier chunked controller restarted training and lost the in-memory replay buffer at every chunk.

`python/azsnek/autotune.py` now mainly provides:

- testable tuning rules (`TuneSettings`, `TuneLimits`, `tune_next`)
- a launcher that starts one long `azsnek.train --adaptive` process

The adaptive policy is conservative:

- starts from lower training pressure (`ADAPTIVE_TRAIN_STEPS ?= 256`)
- caps `train_steps` by replay-buffer epochs
- cuts optimization pressure when policy/value loss regress
- increases fresh samples and eval games around plateaus
- adjusts `tau` only when targets are clearly too soft or too sharp

Useful adaptive command:

```bash
make adaptive RUN_ID=adaptive-tau30 TOTAL_GENERATIONS=100000
```

Tuning cadence is controlled with `ADAPTIVE_EVERY` default `4`.

## Current Training Interpretation

- `policy_loss` is cross-entropy to search policy targets. It cannot go below the entropy of those targets.
- `value_loss` is MSE from predicted value to final game outcome.
- `target_entropy` measures how spread out the search target policy is.
- `target_max_prob` measures how sharp the search target is on average.
- Very high target entropy means targets are too soft/random. Very low entropy with a weak value net can mean overconfident bad targets.
- Recent evidence suggested `TRAIN_STEPS=1024` was too much per generation: it produced millions of sampled updates per gen, policy loss bottomed then backed up, and value loss rose. Adaptive defaults now start lower.

## Dashboard Notes

- Rebuild UI assets with `make ui` after React/CSS changes.
- The metrics graph supports hover tooltips with raw values.
- Replay tiles have scrub controls; scrubbing a tile pauses that tile's autoplay until resumed.
