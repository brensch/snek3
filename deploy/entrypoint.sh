#!/usr/bin/env bash
# Container entrypoint: everything is already built into the image, so just
# launch the trainer-server. Configure via env:
#   SNEK_SERVE_TOKEN  bearer token for write/control/resume (required for control)
#   SNEK_RUN_ID       optional: auto-start a run with this name (omit = idle)
#   SNEK_PORT         dashboard/API port (default 8050)
#   SNEK_COUNT/SNEK_SAMPLES  optional self-play overrides (defaults 64 / 8000)
set -euo pipefail
cd /app

args=(--serve-host 0.0.0.0 --serve-port "${SNEK_PORT:-8050}"
      --num-snakes 2 --count "${SNEK_COUNT:-64}" --samples "${SNEK_SAMPLES:-8000}"
      --depth 2 --iters 120 --tau-min 0.5 --tau-max 10.0 --response-tau 12.0
      --response-after 0 --draw-value -0.9 --exploration-prob 0.15 --max-turns 0
      --eval-batch-size 2048 --filters 64 --blocks 6 --lr 1e-3
      --train-steps 128 --recency 2.0 --batch-size 1024 --buffer-size 500000
      --eval-every 5 --eval-games 64 --record-games 2 --record-every 1)
[ -n "${SNEK_RUN_ID:-}" ] && args+=(--run-id "$SNEK_RUN_ID")

exec .venv/bin/python -m azsnek.train_albatross "${args[@]}"
