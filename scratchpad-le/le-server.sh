#!/usr/bin/env bash
# Launch the snek3 LE trainer/API server with the libtorch runtime env.
# Idempotent-ish: assumes port 8050 is already free (caller waits for it).
set -euo pipefail
cd /home/brensch/snek3

TORCH_SITE=/home/brensch/snek3/.venv/lib/python3.12/site-packages
TORCH_LIB="$TORCH_SITE/torch/lib"
NV=$(find "$TORCH_SITE/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')

export LD_PRELOAD="$TORCH_LIB/libtorch_global_deps.so:$TORCH_LIB/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$TORCH_LIB:$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SNEK_NO_PUBLISH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUST_LOG="snek=info,warn"

exec ./crates/snek-train/target/release/snek-train \
  --bind 127.0.0.1:8050 --runs-dir /home/brensch/snek3/runs
