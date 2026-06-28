#!/bin/sh
# Resolve the CPU onnxruntime shared lib (pip installs it with a version suffix,
# e.g. libonnxruntime.so.1.22.0) for ort's `load-dynamic`, then launch the API.
set -e
export ORT_DYLIB_PATH="$(ls /usr/local/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so* 2>/dev/null | head -1)"
if [ -z "$ORT_DYLIB_PATH" ]; then
  echo "FATAL: libonnxruntime.so not found" >&2
  exit 1
fi
exec /app/snek-server
