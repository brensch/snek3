# CPU-only Battlesnake /move API image: the tch/libtorch snek-server with the
# tracked serving checkpoint baked in. Built from the repo root:
#
#   docker build -f deploy/server.Dockerfile -t snek3-api .
#
# libtorch comes from the torch==2.11.0+cpu wheel — the exact version
# torch-sys 0.24 pins, so no version bypass is needed.

# --- viewer: the embedded /app SPA (rust-embed folds viewer/dist into the binary)
FROM node:22-slim AS viewer
WORKDIR /viewer
COPY crates/snek-server/viewer/package.json crates/snek-server/viewer/package-lock.json ./
RUN npm ci
COPY crates/snek-server/viewer .
RUN npm run build

# --- builder: compile snek-server against the CPU libtorch from the torch wheel
FROM rust:1-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*
# Old pips mis-handle the PyTorch index's package-name normalization
# (typing_extensions vs typing-extensions), so upgrade pip first.
RUN python3 -m venv /opt/torch-venv \
    && /opt/torch-venv/bin/pip install --no-cache-dir -U pip \
    && /opt/torch-venv/bin/pip install --no-cache-dir \
       torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
RUN ln -s "$(/opt/torch-venv/bin/python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')" /opt/torch
ENV PATH=/opt/torch-venv/bin:$PATH \
    LIBTORCH_USE_PYTORCH=1 \
    SNEK_TORCH_DIR=/opt/torch
WORKDIR /src
# snek-server is a detached workspace, but its path deps (snek-core,
# snek-search, snek-tch) resolve through the root workspace manifest.
COPY Cargo.toml Cargo.lock ./
COPY crates crates
COPY --from=viewer /viewer/dist crates/snek-server/viewer/dist
RUN cargo build --release --manifest-path crates/snek-server/Cargo.toml --bin snek-server
# Runtime shared libs, minus the Python bindings the binary never links.
RUN mkdir -p /opt/torch-lib \
    && cp /opt/torch/lib/*.so* /opt/torch-lib/ \
    && rm -f /opt/torch-lib/libtorch_python.so

# --- runtime
FROM debian:bookworm-slim
COPY --from=builder /opt/torch-lib /opt/torch/lib
COPY --from=builder /src/crates/snek-server/target/release/snek-server /app/snek-server
COPY checkpoints/serving.safetensors /app/net.safetensors
COPY checkpoints/serving.json /app/net.json
ENV LD_LIBRARY_PATH=/opt/torch/lib \
    SNEK_MODEL=/app/net.safetensors \
    SNEK_CPU_ONLY=1 \
    SNEK_MOVE_LOG_DIR=/app/logs/api_moves \
    SNEK_PORT=8000 \
    # Arch of the baked checkpoint (see /app/net.json); pinned here so a future
    # default change in the binary can't silently mismatch the weights.
    SNEK_TRUNK_CHANNELS=96 \
    SNEK_TRUNK_BLOCKS=8 \
    SNEK_GPOOL_EVERY=3 \
    # CPU serve tuning, measured on the i5-1340P deploy box: forwards want the
    # 4 P-cores, and batch cost is ~linear so small leaf batches search better.
    SNEK_TORCH_THREADS=4 \
    SNEK_LEAVES_PER_SIM=2
EXPOSE 8000
WORKDIR /app
CMD ["/app/snek-server"]
