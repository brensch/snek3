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
# The root workspace manifest is required because snek-core/snek-search inherit
# `edition`/`version`/deps via `workspace = true`; without it cargo can't find a
# workspace root and the build fails.
COPY Cargo.toml ./Cargo.toml
COPY crates/ ./crates/
RUN cargo build --release --manifest-path crates/snek-server/Cargo.toml

# ---- runtime ----
FROM python:3.12-slim AS runtime
RUN pip install --no-cache-dir onnxruntime
WORKDIR /app
COPY --from=build /src/crates/snek-server/target/release/snek-server /app/snek-server
COPY model.onnx /app/model.onnx
RUN python -c "import glob, os; p=glob.glob('/usr/local/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*')[0]; os.symlink(p, '/usr/local/lib/libonnxruntime.so')"
ENV SNEK_MODEL=/app/model.onnx SNEK_PORT=8000 SNEK_CPU_ONLY=1 ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
EXPOSE 8000
CMD ["/app/snek-server"]
