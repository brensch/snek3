#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_ID="${RUN_ID:-adaptive-$(date +%Y%m%d-%H%M%S)}"
TOTAL_GENERATIONS="${TOTAL_GENERATIONS:-2500}"
CHUNK_GENERATIONS="${CHUNK_GENERATIONS:-4}"
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
RECORD_GAMES="${RECORD_GAMES:-8}"
RECORD_EVERY="${RECORD_EVERY:-5}"
LOG_DIR="${LOG_DIR:-logs}"
FRESH="${FRESH:-}"
ARGS="${ARGS:-}"

mkdir -p "$LOG_DIR"
log="$LOG_DIR/${RUN_ID}.adaptive.log"
pid_file="$LOG_DIR/${RUN_ID}.adaptive.pid"

if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "adaptive run already active: $RUN_ID pid $(cat "$pid_file")"
  exit 1
fi

cmd=(
  .venv/bin/python -m azsnek.autotune
  --run-id "$RUN_ID"
  --total-generations "$TOTAL_GENERATIONS"
  --chunk-generations "$CHUNK_GENERATIONS"
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
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

nohup "${cmd[@]}" >"$log" 2>&1 &
pid="$!"
echo "$pid" > "$pid_file"
echo "pid: $pid"
echo "watch: tail -f $log"
