# Agent Notes

## Repo Shape

- Rust rules/search bindings live under `crates/`; build them into Python with `make build` (`maturin develop --release`).
- Python training code lives in `python/azsnek/`.
- The Battlesnake `/move` API is the pure-Rust `crates/snek-server` (same MCTS as self-play).
- The dashboard backend is in `python/dashboard/`; the React UI source is `python/dashboard/ui/`, with committed built assets in `python/dashboard/static/`.
- Training outputs are runtime data under `runs/` and are ignored by git. Logs are under `logs/` and ignored.

## Common Commands

- `make build`: compile the Rust Python extension.
- `make test-py`: build and run Python tests.
- `make test`: run Rust and Python tests.
- `make train`: foreground fixed-parameter training.
- `make dashboard`: serve the training dashboard on `PORT` default `8050`.

## Running Rust ONNX binaries on the GPU

The `snek-infer` `Net` (used by `bench_batches`, `snek-server`, and Rust self-play)
loads via `ort` with `load-dynamic`, so it needs two env vars to use the GPU.
**If you skip them it silently falls back to CPU** — there is no error; the only
tell is ~4k rows/s and `nvidia-smi` showing 0% util / no extra GPU memory.

When self-play runs via `make train`, Python imports torch first, which pulls the
CUDA runtime libs into the process, so the CUDA EP resolves. A standalone Rust
binary launched from a bare shell has none of that on its loader path, so the
CUDA provider (`libonnxruntime_providers_cuda.so`) can't resolve its deps
(`libcublas`, `libcublasLt`, `libcurand`, `libcufft`, `libcudart`, `libcudnn`,
`libnvrtc`) and `ort` quietly uses CPU. Fix by putting **every** `nvidia/*/lib`
dir on `LD_LIBRARY_PATH`:

```sh
SP=.venv/lib/python3.12/site-packages
export ORT_DYLIB_PATH="$(ls $SP/onnxruntime/capi/libonnxruntime.so* | head -1)"
export LD_LIBRARY_PATH="$(find $SP/nvidia -name lib -type d | tr '\n' ':')$SP/onnxruntime/capi:$LD_LIBRARY_PATH"
```

Verify with `nvidia-smi` (util should jump, GPU mem grows by a few GB). On the
RTX 5080 this net (`[N,14,11,11]` → policy[4]+value) does ~50k rows/s at batch
512. The `crates/snek-infer/examples/bench_batches.rs` batch-size sweep needs
`SNEK_BENCH_C=14 SNEK_BENCH_H=11 SNEK_BENCH_W=11` to match the real net shape.

## Training State

- Each run writes to `runs/<run-id>/`.
- `state.pt` is the full resumable training state: network, optimizer, RNG, generation, and best win rate.
- Serving checkpoints are per-run in `runs/<run-id>/ckpt/latest.pt` and `best.pt`.
- `metrics.jsonl` is one JSON object per generation and feeds the dashboard.
- `meta.json` records run config and may be updated by adaptive tuning.
- The replay buffer is currently in memory only. Restarting `azsnek.train` resumes the net/optimizer but does not restore the replay buffer.

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
