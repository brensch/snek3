#!/usr/bin/env bash
# Trainer container entrypoint: bring up the tailnet tunnel if TS_AUTHKEY is
# set, then exec the trainer. Extra args are passed through to snek-train.
#
# Env: TS_AUTHKEY, TS_HOSTNAME (see tailscale-dashboard.sh), BIND, RUNS_DIR,
#      RUN_ID, FRESH, START (non-empty = begin training immediately).
set -euo pipefail

BIND="${BIND:-127.0.0.1:8050}"
export DASH_PORT="${DASH_PORT:-${BIND##*:}}"

# Backgrounded so a slow or blocked tunnel (e.g. `tailscale serve` waiting for
# the feature to be enabled on the tailnet) never holds up training.
if [[ -n "${TS_AUTHKEY:-}" ]]; then
    (/app/tailscale-dashboard.sh || echo "warning: tailscale tunnel failed; dashboard only on $BIND" >&2) &
else
    echo "TS_AUTHKEY not set: no tailnet tunnel; dashboard only on $BIND" >&2
fi

# tch loads CUDA lazily; preloading forces libtorch_cuda's device registration
# (same trick as the Makefile's LIBTORCH_PRELOAD). Scoped to the trainer exec
# so tailscaled above doesn't map libtorch.
TORCH_LIB=/opt/libtorch/lib
if [[ -f "$TORCH_LIB/libtorch_cuda.so" ]]; then
    export LD_PRELOAD="$TORCH_LIB/libtorch_global_deps.so:$TORCH_LIB/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

exec /app/snek-train \
    --bind "$BIND" \
    --runs-dir "${RUNS_DIR:-/runs}" \
    ${RUN_ID:+--run-id "$RUN_ID"} \
    ${FRESH:+--fresh} \
    ${START:+--start} \
    "$@"
