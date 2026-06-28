# snek3 trainer-server — fully precompiled image (torch + Rust extension baked
# in, so the pod boots straight into serving; no per-boot build).
#
# Built in CI -> ghcr.io/brensch/snek3:latest (see .github/workflows/docker.yml).
# Run:  docker run --gpus all -p 8050:8050 -e SNEK_SERVE_TOKEN=secret \
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
COPY . /app
# .venv with torch + deps, then compile the Rust extension into it.
RUN make venv && make build && chmod +x deploy/entrypoint.sh

EXPOSE 8050
CMD ["/app/deploy/entrypoint.sh"]
