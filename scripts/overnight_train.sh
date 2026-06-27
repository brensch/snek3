#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAU="${TAU:-30}"
GENERATIONS="${GENERATIONS:-2500}"
SAMPLES="${SAMPLES:-50000}"
COUNT="${COUNT:-32}"
DEPTH="${DEPTH:-2}"
ITERS="${ITERS:-120}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8192}"
SEARCH_THREADS="${SEARCH_THREADS:-$(nproc 2>/dev/null || python3 -c 'import os; print(os.cpu_count() or 1)')}"
TRAIN_STEPS="${TRAIN_STEPS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
BUFFER_SIZE="${BUFFER_SIZE:-500000}"
FILTERS="${FILTERS:-64}"
BLOCKS="${BLOCKS:-6}"
EVAL_EVERY="${EVAL_EVERY:-1}"
EVAL_GAMES="${EVAL_GAMES:-32}"
MAX_TURNS="${MAX_TURNS:-0}"
SAMPLE_GAMES="${SAMPLE_GAMES:-16}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
RECORD_GAMES="${RECORD_GAMES:-8}"
RECORD_EVERY="${RECORD_EVERY:-5}"
RUN_ID="${RUN_ID:-overnight-tau${TAU}-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs}"
FRESH="${FRESH:-}"

mkdir -p "$LOG_DIR"
log="$LOG_DIR/${RUN_ID}.log"
pid_file="$LOG_DIR/${RUN_ID}.pid"

if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "run already active: $RUN_ID pid $(cat "$pid_file")"
  exit 1
fi

cmd=(
  .venv/bin/python -m azsnek.train
  --generations "$GENERATIONS"
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
  --eval-every "$EVAL_EVERY"
  --eval-games "$EVAL_GAMES"
  --max-turns "$MAX_TURNS"
  --sample-games "$SAMPLE_GAMES"
  --sample-every "$SAMPLE_EVERY"
  --record-games "$RECORD_GAMES"
  --record-every "$RECORD_EVERY"
  --run-id "$RUN_ID"
)

if [[ -n "$FRESH" ]]; then
  cmd+=(--fresh)
fi

echo "starting $RUN_ID"
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
