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
# libtorch comes from the torch==2.11.0+cu128 wheel — the exact version
# torch-sys 0.24 pins, so no version bypass is needed.

# --- frontend: the dashboard SPA the trainer serves at /
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend .
RUN npm run build

# --- builder: compile snek-train against the CUDA libtorch from the torch wheel
FROM rust:1-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
# Old pips mis-handle the PyTorch index's package-name normalization
# (typing_extensions vs typing-extensions), so upgrade pip first.
RUN python3 -m venv /opt/torch-venv \
    && /opt/torch-venv/bin/pip install --no-cache-dir -U pip \
    && /opt/torch-venv/bin/pip install --no-cache-dir \
       torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# The snek-tch shim also reads /opt/torch/../triton for the CUDA include tree;
# the kernel resolves that through this symlink into the venv's site-packages.
RUN ln -s "$(/opt/torch-venv/bin/python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')" /opt/torch
ENV PATH=/opt/torch-venv/bin:$PATH \
    LIBTORCH_USE_PYTORCH=1 \
    SNEK_TORCH_DIR=/opt/torch
WORKDIR /src
# snek-train is a detached workspace, but its path deps (snek-core, snek-tch)
# resolve through the root workspace manifest.
COPY Cargo.toml Cargo.lock ./
COPY crates crates
RUN cargo build --release --manifest-path crates/snek-train/Cargo.toml
# Runtime shared libs with the wheel layout preserved: torch/lib's RPATHs
# ($ORIGIN/../../nvidia/*/lib) locate the CUDA libs, so LD_LIBRARY_PATH only
# needs torch/lib itself.
RUN SP="$(/opt/torch-venv/bin/python3 -c 'import site; print(site.getsitepackages()[0])')" \
    && mkdir -p /opt/site/torch \
    && cp -a "$SP/torch/lib" /opt/site/torch/lib \
    && cp -a "$SP/nvidia" /opt/site/nvidia \
    && rm -f /opt/site/torch/lib/libtorch_python.so

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
COPY --from=builder /opt/site /opt/site
COPY --from=builder /src/crates/snek-train/target/release/snek-train /app/snek-train
COPY --from=frontend /frontend/dist /app/frontend/dist
COPY deploy/tailscale-dashboard.sh deploy/trainer-entrypoint.sh /app/
ENV LD_LIBRARY_PATH=/opt/site/torch/lib \
    # tch loads CUDA lazily; preloading forces libtorch_cuda's device
    # registration (same trick as the Makefile's LIBTORCH_PRELOAD).
    LD_PRELOAD=/opt/site/torch/lib/libtorch_global_deps.so:/opt/site/torch/lib/libtorch_cuda.so \
    RUNS_DIR=/runs
# The dashboard binds 127.0.0.1 by default (tailnet-only via the tunnel).
# Set BIND=0.0.0.0:8050 to reach it through a mapped port instead.
EXPOSE 8050
WORKDIR /app
ENTRYPOINT ["/app/trainer-entrypoint.sh"]
