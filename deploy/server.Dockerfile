# CPU-only Battlesnake /move API image: the tch/libtorch snek-server with the
# tracked serving checkpoint baked in. Built from the repo root:
#
#   docker build -f deploy/server.Dockerfile -t snek3-api .
#
# libtorch comes from PyTorch's official python-free C++ bundle
# (libtorch-shared-with-deps, cpu) — the exact 2.11.0 torch-sys 0.24 pins.
# No Python in any stage.

# --- viewer: the embedded /app SPA (rust-embed folds viewer/dist into the binary)
FROM node:22-slim AS viewer
WORKDIR /viewer
COPY crates/snek-server/viewer/package.json crates/snek-server/viewer/package-lock.json ./
RUN npm ci
COPY crates/snek-server/viewer .
RUN npm run build

# --- builder: compile snek-server against the CPU libtorch bundle
FROM rust:1-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends unzip \
    && rm -rf /var/lib/apt/lists/*
# No libtorch_cuda.so in the CPU bundle, so the snek-tch CUDA-graph shim is
# compiled out (no CUDA headers needed). Python bindings dropped up front.
RUN curl -fsSL -o /tmp/libtorch.zip \
        "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.11.0%2Bcpu.zip" \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip /opt/libtorch/lib/libtorch_python.so
ENV LIBTORCH=/opt/libtorch \
    SNEK_TORCH_DIR=/opt/libtorch
WORKDIR /src
# snek-server is a detached workspace, but its path deps (snek-core,
# snek-search, snek-tch) resolve through the root workspace manifest.
COPY Cargo.toml Cargo.lock ./
COPY crates crates
COPY --from=viewer /viewer/dist crates/snek-server/viewer/dist
# Cache mounts keep the dep graph compiled across builds; only changed crates
# rebuild. The binary is copied out because the target dir isn't in the layer.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/crates/snek-server/target \
    cargo build --release --manifest-path crates/snek-server/Cargo.toml --bin snek-server \
    && cp crates/snek-server/target/release/snek-server /usr/local/bin/snek-server

# --- runtime
FROM debian:bookworm-slim
COPY --from=builder /opt/libtorch/lib /opt/torch/lib
COPY --from=builder /usr/local/bin/snek-server /app/snek-server
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
    # CPU serve tuning, measured on the i5-1340P deploy box: forwards want the
    # 4 P-cores, and batch cost is ~linear so small leaf batches search better.
    SNEK_TORCH_THREADS=4 \
    SNEK_LEAVES_PER_SIM=2
EXPOSE 8000
WORKDIR /app
CMD ["/app/snek-server"]
