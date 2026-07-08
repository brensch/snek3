#!/usr/bin/env bash
# Publish the newest rated checkpoint as the serving net and rebuild the API
# container.
#
#   deploy/publish-serving.sh          publish once (if there's a new entrant)
#   deploy/publish-serving.sh --watch  poll forever, publishing as they land
#
# What "publish" means: overwrite checkpoints/serving.safetensors on a
# single-commit `serving` branch (origin/main + weights, force-pushed) and
# `gh workflow run api-image.yml --ref serving` so ghcr.io/…/snek3-api:latest
# is rebuilt with the new net baked in. The weights never enter main's
# history — a 5.6MB safetensors per generation would bloat the repo by
# gigabytes within weeks, so the branch is recreated from origin/main on
# every publish and holds exactly one weights commit.
#
# Which net: the newest checkpoint in runs/<run>/checkpoints whose generation
# is a multiple of the run's league_entrant_gens — i.e. the latest net that
# actually holds a league rating, at the same cadence the arena rates them.
#
# Env:
#   RUN_ID    run to publish from (default: the trainer API's active run)
#   API       trainer API base (default http://127.0.0.1:8050)
#   INTERVAL  watch-mode poll seconds (default 600)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API="${API:-http://127.0.0.1:8050}"
INTERVAL="${INTERVAL:-600}"
BRANCH=serving

active_run() {
    if [[ -n "${RUN_ID:-}" ]]; then printf '%s' "$RUN_ID"; return; fi
    curl -fsS -m 5 "$API/api/state" | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])'
}

# Newest checkpoint gen that is a league entrant (multiple of entrant_gens).
latest_entrant() { # latest_entrant <run-dir> -> gen number or empty
    local dir=$1 entrant
    entrant=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("league_entrant_gens",20) or 20)' \
        "$dir/config.json" 2>/dev/null || echo 20)
    ls "$dir/checkpoints" 2>/dev/null |
        sed -n 's/^net_\([0-9]*\)\.safetensors$/\1/p' |
        awk -v e="$entrant" 'int($0) % e == 0' | sort -n | tail -1
}

published() { # "run gen" recorded in the serving branch's commit subject, or empty
    git -C "$REPO_DIR" fetch -q origin "$BRANCH" 2>/dev/null || return 0
    git -C "$REPO_DIR" log -1 --format=%s "origin/$BRANCH" 2>/dev/null |
        sed -n 's/^serve: \(.*\) gen \([0-9]*\)$/\1 \2/p'
}

publish() { # publish <run> <gen>
    local run=$1 gen=$2
    local src="$REPO_DIR/runs/$run/checkpoints/net_$(printf '%04d' "$gen").safetensors"
    [[ -f "$src" ]] || { echo "error: $src vanished" >&2; return 1; }
    local wt
    wt=$(mktemp -d)
    git -C "$REPO_DIR" fetch -q origin main
    git -C "$REPO_DIR" worktree add -q --detach "$wt" origin/main
    (
        cd "$wt"
        mkdir -p checkpoints
        cp "$src" checkpoints/serving.safetensors
        python3 - "$run" "$gen" "$REPO_DIR/runs/$run/config.json" <<'PY'
import hashlib, json, sys, datetime
run, gen, cfg_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = json.load(open(cfg_path))
blob = open("checkpoints/serving.safetensors", "rb").read()
json.dump({
    "source_run": run,
    "generation": gen,
    "exported": datetime.date.today().isoformat(),
    "sha256": hashlib.sha256(blob).hexdigest(),
    "trunk_channels": cfg["trunk_channels"],
    "trunk_blocks": cfg["trunk_blocks"],
    "gpool_every": 3,
    "note": "auto-published by deploy/publish-serving.sh; must ship with matching crates/snek-tch",
}, open("checkpoints/serving.json", "w"), indent=2)
PY
        git checkout -q -B "$BRANCH"
        git add checkpoints/serving.safetensors checkpoints/serving.json
        git commit -q -m "serve: $run gen $gen"
        git push -q -f origin "$BRANCH"
    )
    git -C "$REPO_DIR" worktree remove -f "$wt"
    if command -v gh >/dev/null; then
        gh workflow run api-image.yml --ref "$BRANCH" -R "$(git -C "$REPO_DIR" remote get-url origin | sed 's/.*github.com[:/]//;s/\.git$//')" \
            && echo "publish: $run gen $gen pushed; api-image build dispatched" \
            || echo "publish: $run gen $gen pushed; api-image dispatch FAILED (run it manually)" >&2
    else
        echo "publish: $run gen $gen pushed; no gh CLI — dispatch api-image manually" >&2
    fi
}

cycle() {
    local run gen last
    run=$(active_run) || { echo "no active run (trainer API down?); skipping" >&2; return 0; }
    gen=$(latest_entrant "$REPO_DIR/runs/$run")
    [[ -n "$gen" ]] || { echo "no entrant checkpoints in runs/$run yet" >&2; return 0; }
    gen=$((10#$gen)) # zero-padded filename digits; force base-10 (0560 is NOT octal)
    last=$(published)
    if [[ "$run $gen" == "$last" ]]; then
        echo "$run gen $gen already published"
    else
        publish "$run" "$gen"
    fi
}

if [[ "${1:-}" == "--watch" ]]; then
    echo "watching for new entrant checkpoints every ${INTERVAL}s..."
    while true; do
        cycle || echo "publish cycle failed; retrying next interval" >&2
        sleep "$INTERVAL"
    done
else
    cycle
fi
