# snek3 trainer-server container. GPU passthrough via the NVIDIA Container
# Toolkit (--gpus all) runs CUDA at native speed — no GPU perf loss.
#
# Build:  docker build -t snek3 -f deploy/Dockerfile .
# Run:    docker run --gpus all -p 8050:8050 \
#           -e SNEK_SERVE_TOKEN=CHANGEME \
#           -v snek-runs:/app/runs snek3
#         (the -v keeps runs/ checkpoints across container restarts)
#
# The torch wheel (cu128) bundles its own CUDA runtime, so only the host NVIDIA
# driver is needed — hence a plain python base works.
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
RUN make venv && make build

EXPOSE 8050
# Idle server (create runs from the dashboard). Pass --run-id <name> to autostart.
CMD ["/app/.venv/bin/python", "-m", "azsnek.train_albatross", \
     "--serve-host", "0.0.0.0", "--serve-port", "8050", \
     "--num-snakes", "2", "--count", "64", "--samples", "8000", "--depth", "2", \
     "--draw-value", "-0.9", "--filters", "64", "--blocks", "6", \
     "--train-steps", "128", "--recency", "2.0", "--eval-every", "5", \
     "--eval-games", "64", "--response-after", "0"]
