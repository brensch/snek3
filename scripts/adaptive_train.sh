#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_ID="${RUN_ID:-adaptive-$(date +%Y%m%d-%H%M%S)}"
TOTAL_GENERATIONS="${TOTAL_GENERATIONS:-2500}"
ADAPTIVE_EVERY="${ADAPTIVE_EVERY:-${CHUNK_GENERATIONS:-4}}"
SAMPLES="${SAMPLES:-50000}"
COUNT="${COUNT:-32}"
DEPTH="${DEPTH:-2}"
TAU="${TAU:-30}"
ITERS="${ITERS:-120}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8192}"
SEARCH_THREADS="${SEARCH_THREADS:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
BUFFER_SIZE="${BUFFER_SIZE:-500000}"
FILTERS="${FILTERS:-64}"
BLOCKS="${BLOCKS:-6}"
EVAL_GAMES="${EVAL_GAMES:-64}"
MAX_TURNS="${MAX_TURNS:-0}"
EXPLORATION_PROB="${EXPLORATION_PROB:-0.15}"
DRAW_VALUE="${DRAW_VALUE:--0.25}"
SKIP_SHORT_DRAW_TURNS="${SKIP_SHORT_DRAW_TURNS:-0}"
SAMPLE_GAMES="${SAMPLE_GAMES:-16}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
RECORD_GAMES="${RECORD_GAMES:-8}"
RECORD_EVERY="${RECORD_EVERY:-5}"
LOG_DIR="${LOG_DIR:-logs}"
FRESH="${FRESH:-}"
ARGS="${ARGS:-}"

mkdir -p "$LOG_DIR"
log="$LOG_DIR/${RUN_ID}.adaptive.log"

cmd=(
  .venv/bin/python -m azsnek.autotune
  --run-id "$RUN_ID"
  --total-generations "$TOTAL_GENERATIONS"
  --adaptive-every "$ADAPTIVE_EVERY"
  --samples "$SAMPLES"
  --count "$COUNT"
  --depth "$DEPTH"
  --tau "$TAU"
  --iters "$ITERS"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --search-threads "$SEARCH_THREADS"
  --train-steps "$TRAIN_STEPS"
  --batch-size "$BATCH_SIZE"
  --buffer-size "$BUFFER_SIZE"
  --filters "$FILTERS"
  --blocks "$BLOCKS"
  --eval-games "$EVAL_GAMES"
  --max-turns "$MAX_TURNS"
  --exploration-prob "$EXPLORATION_PROB"
  --draw-value "$DRAW_VALUE"
  --skip-short-draw-turns "$SKIP_SHORT_DRAW_TURNS"
  --sample-games "$SAMPLE_GAMES"
  --sample-every "$SAMPLE_EVERY"
  --record-games "$RECORD_GAMES"
  --record-every "$RECORD_EVERY"
)

if [[ -n "$FRESH" ]]; then
  cmd+=(--fresh)
fi

if [[ -n "$ARGS" ]]; then
  # shellcheck disable=SC2206
  extra=($ARGS)
  cmd+=("${extra[@]}")
fi

echo "starting adaptive $RUN_ID"
echo "log: $log"
echo "dashboard run dir: runs/$RUN_ID"
echo "press Ctrl-C to stop"
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee -a "$log"
