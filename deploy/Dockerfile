# snek3 trainer-server — fully precompiled image (torch + Rust extension baked
# in, so the pod boots straight into serving; no per-boot build).
#
# Built in CI -> ghcr.io/brensch/snek3:latest (see .github/workflows/docker.yml).
# Run:  docker run --gpus all -p 8050:8050 \
#         -e SNEK_RUN_ID=h100-test ghcr.io/brensch/snek3:latest
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
RUN .venv/bin/maturin develop --release && chmod +x deploy/entrypoint.sh

EXPOSE 8050
CMD ["/app/deploy/entrypoint.sh"]
