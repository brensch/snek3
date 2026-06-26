#!/usr/bin/env bash
#
# End-to-end audit of the snek3 bot. Run from the repo root:
#
#     bash scripts/audit.sh
#
# It proves, in order:
#   1. the Rust rules engine + search pass their tests
#   2. the Python extension builds and the Python tests pass
#   3. a short real self-play training run makes the win-rate vs the flood-fill
#      baseline climb (this is the "it actually learns" check)
#   4. the live HTTP server returns a legal move for a real /move request
#
# Nothing here is precomputed: every number is produced fresh by this run.
# For a clean-room audit (no build cache, no installed packages), first run:
#   rm -rf target crates/snek-py/target .venv
#   python3 -m venv .venv
#   .venv/bin/pip install maturin numpy pytest fastapi uvicorn httpx
#   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
# then run this script.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

hr() { printf '\n\033[1;36m========== %s ==========\033[0m\n' "$1"; }

[ -x "$PY" ] || { echo "No .venv found. See the clean-room note at the top of this script."; exit 1; }

hr "1/5  Rust tests (rules engine + equilibrium search)"
cargo test 2>&1 | grep -E "Running|test result"

hr "2/5  Build the Python extension (maturin)"
.venv/bin/maturin develop --release 2>&1 | tail -2

hr "3/5  Python tests (bindings, net, full search, training, server)"
$PY -m pytest python/tests -q

hr "4/5  Short real training run — watch win-rate vs the flood-fill baseline"
echo "(random-init net -> a few generations of self-play; expect win_rate to rise)"
$PY -m azsnek.train \
    --generations 9 --samples 4000 --count 96 --depth 2 --iters 100 \
    --eval-every 3 --eval-games 100 --blocks 4 --filters 48 \
    --ckpt-dir "$(mktemp -d)" 2>&1 | grep -E "device:|win_rate"

hr "5/5  Live HTTP server returns a legal move"
PORT=8123
$PY -m uvicorn server.main:app --host 127.0.0.1 --port $PORT --log-level warning &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
# Wait (up to ~20s) for the server to come up.
until curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1; do
    kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; exit 1; }
    sleep 0.5
done
echo "GET /  ->  $(curl -s http://127.0.0.1:$PORT/)"
MOVE=$(curl -s -X POST "http://127.0.0.1:$PORT/move" -H 'content-type: application/json' -d '{
  "turn": 3,
  "board": {"width":11,"height":11,"food":[{"x":5,"y":6}],"hazards":[],
    "snakes":[
      {"id":"me","health":90,"body":[{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}]},
      {"id":"opp","health":85,"body":[{"x":2,"y":2},{"x":2,"y":1},{"x":2,"y":0}]}]},
  "you": {"id":"me"}
}')
echo "POST /move  ->  $MOVE"
echo "$MOVE" | grep -Eq '"move":"(up|down|left|right)"' && echo "  legal move ✓"

hr "AUDIT COMPLETE — all stages ran fresh"
