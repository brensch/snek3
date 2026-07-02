# CUDA trainer image: snek-train with the dashboard frontend and a userspace
# Tailscale tunnel baked in. Built from the repo root:
#
#   docker build -f deploy/trainer.Dockerfile -t snek3-trainer .
#
# Run on any NVIDIA-runtime pod:
#
#   docker run --gpus all -v /your/volume:/runs \
#     -e TS_AUTHKEY=tskey-auth-... -e START=1 -e RUN_ID=main snek3-trainer
#
# libtorch comes from PyTorch's official python-free C++ bundle
# (libtorch-shared-with-deps, cu128) — the exact 2.11.0 torch-sys 0.24 pins,
# with all CUDA runtime libs (cudart/cublas/cudnn/...) included flat in lib/.
# No Python in any stage.

# --- frontend: the dashboard SPA the trainer serves at /
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend .
RUN npm run build

# --- cuda-headers: only /usr/local/cuda/include is taken from this stage; the
# snek-tch CUDA-graph shim needs host-compilable CUDA headers, which the
# libtorch bundle doesn't ship.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS cuda-headers

# --- builder: compile snek-train against the libtorch bundle
FROM rust:1-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler unzip \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/libtorch.zip \
        "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.11.0%2Bcu128.zip" \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip /opt/libtorch/lib/libtorch_python.so
COPY --from=cuda-headers /usr/local/cuda/include /opt/cuda/include
ENV LIBTORCH=/opt/libtorch \
    SNEK_TORCH_DIR=/opt/libtorch \
    SNEK_CUDA_INC=/opt/cuda/include
WORKDIR /src
# snek-train is a detached workspace, but its path deps (snek-core, snek-tch)
# resolve through the root workspace manifest.
COPY Cargo.toml Cargo.lock ./
COPY crates crates
# Cache mounts keep the dep graph compiled across builds; only changed crates
# rebuild. The binary is copied out because the target dir isn't in the layer.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/crates/snek-train/target \
    cargo build --release --manifest-path crates/snek-train/Cargo.toml \
    && cp crates/snek-train/target/release/snek-train /usr/local/bin/snek-train

# --- runtime
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
# Pre-bake static tailscale binaries so pods don't download them at boot.
RUN set -eu; \
    case "$(uname -m)" in x86_64) arch=amd64 ;; aarch64) arch=arm64 ;; esac; \
    tarball="$(curl -fsSL 'https://pkgs.tailscale.com/stable/?mode=json' \
        | grep -o "\"tailscale_[0-9.]*_${arch}.tgz\"" | head -1 | tr -d '"')"; \
    curl -fsSL "https://pkgs.tailscale.com/stable/$tarball" \
        | tar -xz -C /usr/local/bin --strip-components=1 \
            --wildcards '*/tailscale' '*/tailscaled'
COPY --from=builder /opt/libtorch/lib /opt/libtorch/lib
COPY --from=builder /usr/local/bin/snek-train /app/snek-train
COPY --from=frontend /frontend/dist /app/frontend/dist
COPY deploy/tailscale-dashboard.sh deploy/trainer-entrypoint.sh /app/
ENV LD_LIBRARY_PATH=/opt/libtorch/lib \
    RUNS_DIR=/runs
# The dashboard binds 127.0.0.1 by default (tailnet-only via the tunnel).
# Set BIND=0.0.0.0:8050 to reach it through a mapped port instead.
EXPOSE 8050
WORKDIR /app
ENTRYPOINT ["/app/trainer-entrypoint.sh"]
