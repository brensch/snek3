#!/usr/bin/env bash
# Trainer container entrypoint: bring up the tailnet tunnel if TS_AUTHKEY is
# set, then exec the trainer. Extra args are passed through to snek-train.
#
# Env: TS_AUTHKEY, TS_HOSTNAME (see tailscale-dashboard.sh), BIND, RUNS_DIR,
#      RUN_ID, FRESH, START (non-empty = begin training immediately).
set -euo pipefail

BIND="${BIND:-127.0.0.1:8050}"
export DASH_PORT="${DASH_PORT:-${BIND##*:}}"

if [[ -n "${TS_AUTHKEY:-}" ]]; then
    /app/tailscale-dashboard.sh || echo "warning: tailscale tunnel failed; continuing without it" >&2
else
    echo "TS_AUTHKEY not set: no tailnet tunnel; dashboard only on $BIND" >&2
fi

exec /app/snek-train \
    --bind "$BIND" \
    --runs-dir "${RUNS_DIR:-/runs}" \
    ${RUN_ID:+--run-id "$RUN_ID"} \
    ${FRESH:+--fresh} \
    ${START:+--start} \
    "$@"
