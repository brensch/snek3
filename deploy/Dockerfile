# snek3 trainer-server — fully precompiled image (torch + Rust extension baked
# in, so the pod boots straight into serving; no per-boot build).
#
# Built in CI -> ghcr.io/brensch/snek3:latest (see .github/workflows/trainer-image.yml).
# Run:
#   docker run --gpus all -p 8050:8050 ghcr.io/brensch/snek3:latest --run-id a100-test
#
# torch cu128 wheels bundle their own CUDA runtime, so only a recent host NVIDIA
# driver is needed (--gpus all). Backward-compatible with CUDA 12.8 / 13 hosts.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl build-essential pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain for the pyo3 extension (maturin).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app
# Python deps layer FIRST (torch is ~5GB) so it stays cached across source
# changes — only re-runs if this list changes. Mirrors `make venv`.
RUN python -m venv .venv \
 && .venv/bin/pip install -U pip maturin numpy fastapi uvicorn \
 && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128

# Source + compile the Rust extension (re-runs on code change; ~2-3 min).
COPY . /app
RUN .venv/bin/maturin develop --release \
 && mkdir -p /workspace/runs

EXPOSE 8050
ENTRYPOINT ["/app/.venv/bin/python", "-m", "azsnek.train"]
CMD ["--serve", "--serve-host", "0.0.0.0", "--serve-port", "8050", "--runs-dir", "/workspace/runs", "--generations", "100000", "--board", "11", "--num-snakes", "4", "--count", "1024", "--samples", "30000", "--sims", "32", "--c-puct", "1.5", "--trunk-channels", "96", "--trunk-blocks", "8", "--lr", "1e-3", "--train-steps", "256", "--batch-size", "2048", "--buffer-size", "500000", "--exploration-prob", "0.15", "--draw-value", "-0.25", "--max-turns", "200", "--eval-batch-size", "8192", "--sample-games", "16", "--sample-every", "1", "--keep-games", "40"]
