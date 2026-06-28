# Battlesnake /move API server — pure-Rust, CPU-only, for a small box.
#
# Two stages: build the Rust binary, then a slim runtime that carries only the
# binary + a CPU onnxruntime + the exported proxy model. No PyTorch, no Python in
# the request path (Python is present only to ship the onnxruntime .so that ort
# loads dynamically).
#
# Build (from repo root, with model.onnx already exported next to this file's
# build context — see scripts/export_model.py):
#   docker build -f deploy/server.Dockerfile -t snek-api .
#   docker run -p 8000:8000 snek-api
#
# Tunables (env): SNEK_DEPTH SNEK_ITERS SNEK_RESPONSE_TAU SNEK_DRAW_VALUE
#                 SNEK_THREADS SNEK_PORT

# ---- build ----
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY crates/ ./crates/
RUN cargo build --release --manifest-path crates/snek-server/Cargo.toml

# ---- runtime ----
FROM python:3.12-slim AS runtime
RUN pip install --no-cache-dir onnxruntime
WORKDIR /app
COPY --from=build /src/crates/snek-server/target/release/snek-server /app/snek-server
COPY deploy/server-entrypoint.sh /app/entrypoint.sh
COPY model.onnx /app/model.onnx
RUN chmod +x /app/entrypoint.sh
ENV SNEK_MODEL=/app/model.onnx SNEK_PORT=8000
EXPOSE 8000
CMD ["/app/entrypoint.sh"]
