#!/usr/bin/env bash
# One-shot setup for running the snek3 trainer-server on a fresh rented GPU box
# (RunPod / Lambda / Vast H100, Ubuntu-ish base with NVIDIA driver + CUDA).
#
#   curl -fsSL https://raw.githubusercontent.com/brensch/snek3/albatross-learning-signal/scripts/h100.sh | bash
# or clone first and run:  bash scripts/h100.sh
#
# Idempotent: safe to re-run. Override REPO/BRANCH/DIR via env.
set -euo pipefail

REPO="${REPO:-https://github.com/brensch/snek3.git}"
BRANCH="${BRANCH:-albatross-learning-signal}"
DIR="${DIR:-$HOME/snek3}"

# 1. build deps + Rust toolchain (maturin compiles the pyo3 extension)
if ! command -v cargo >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y && apt-get install -y --no-install-recommends \
      git curl build-essential pkg-config ca-certificates || true
  fi
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"

# 2. repo
if [ ! -d "$DIR/.git" ]; then git clone "$REPO" "$DIR"; fi
cd "$DIR"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only

# 3. venv (torch + deps) + compile the Rust extension
make venv
make build

echo
echo "=== GPU check ==="
.venv/bin/python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(" torch", torch.__version__, "cuda", ok, torch.cuda.get_device_name(0) if ok else "")
PY

# Non-interactive launch (for a Vast/RunPod on-start command): set SNEK_SERVE=1
# and SNEK_SERVE_TOKEN=... in the instance env. Optionally SNEK_RUN_ID=name to
# auto-start a run, SNEK_PORT to change the port. Runs in the foreground with a
# crash-restart loop so the box keeps serving.
if [ "${SNEK_SERVE:-0}" = "1" ]; then
  echo "=== launching server (SNEK_SERVE=1) on :${SNEK_PORT:-8050} ==="
  # shellcheck disable=SC2086
  exec bash -c 'while true; do \
    make server ALB_SERVE_PORT='"${SNEK_PORT:-8050}"' \
      '"$( [ -n "${SNEK_RUN_ID:-}" ] && echo RUN_ID=${SNEK_RUN_ID} )"' ; \
    echo "server exited ($?); restarting in 5s"; sleep 5; \
  done'
fi

cat <<'EOF'

=== ready ===
Start the server (public on :8050, token-protected). Pick your own token:

  make server ALB_SERVE_PORT=8050 ALB_SERVE_TOKEN=CHANGEME RUN_ID=h100-test

  - RUN_ID=h100-test  -> auto-start a fresh run named h100-test
  - omit RUN_ID       -> idle; start a run from the dashboard homepage

Open the dashboard at your box's public URL for port 8050 (e.g. RunPod's
"Connect -> HTTP :8050" proxy URL), paste the token, and watch gen_seconds /
samples_per_sec / GPU% vs your local baseline.

To stop being billed: TERMINATE/DESTROY the instance (not just "stop").
Grab results first if you want them:  scp the runs/h100-test dir down, or
just read the throughput numbers off the dashboard.
EOF
