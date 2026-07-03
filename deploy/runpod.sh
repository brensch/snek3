#!/usr/bin/env bash
# Manage snek3 trainer pods on RunPod.
#
#   deploy/runpod.sh launch [gpu]           create an on-demand pod (default: 4090)
#   deploy/runpod.sh status                 list your pods with cost + GPU
#   deploy/runpod.sh stop <pod-id>          terminate a pod (volume is destroyed too)
#   deploy/runpod.sh sync <run-id> [host]   copy runs/<run-id> onto the pod over Tailscale SSH
#   deploy/runpod.sh pull <run-id> [host]   copy /runs/<run-id> back from the pod
#
# gpu accepts a shorthand (4090, 5090, a100, h100, l40s) or a full RunPod
# gpuTypeId like "NVIDIA GeForce RTX 4090".
#
# Keys are read from the environment, then the repo .env, then ~/.snek-ts.env:
#   RUNPOD_APIKEY   RunPod API key
#   TS_AUTHKEY      tailnet auth key (reusable + ephemeral; see DASHBOARD.md)
#
# Launch knobs (env): RUN_ID (default main), TS_HOSTNAME (default
#   snek-train-runpod), IMAGE (default ghcr.io/brensch/snek3-trainer:latest),
#   DISK_GB (default 20), START=1 to begin training at boot (default: off —
#   seed /runs first, then start from the dashboard).
#
# Storage: by default pods mount the "snek3-runs" network volume (NETVOL,
#   default ucc8x95ndn, US-NC-1) at /runs — it outlives pods, so runs survive
#   termination and every new pod sees the same data. Network volumes are
#   secure-cloud only and pin the pod to their datacenter. Set NETVOL= (empty)
#   for a pod-local volume instead (VOLUME_GB, default 50; CLOUD selects
#   COMMUNITY|SECURE|ALL, default COMMUNITY).
#
# On-demand only, deliberately: spot pods get preempted and the replay buffer
# isn't checkpointed, so every interruption costs ~20-30 generations of
# flat win-rate while the buffer refills.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API=https://api.runpod.io/graphql

env_lookup() { # env_lookup VAR -> value from env, repo .env, or ~/.snek-ts.env
    local var=$1 file val
    val="${!var:-}"
    if [[ -z "$val" ]]; then
        for file in "$REPO_DIR/.env" "$HOME/.snek-ts.env"; do
            [[ -f "$file" ]] || continue
            val=$(grep -m1 "^${var}=" "$file" | cut -d= -f2-) && [[ -n "$val" ]] && break
        done
    fi
    [[ -n "$val" ]] || { echo "error: $var not set (env, .env, ~/.snek-ts.env)" >&2; exit 1; }
    printf '%s' "$val"
}

gql() { # gql <query-json> -> response body, dies on GraphQL errors
    local resp
    resp=$(curl -fsS -X POST -H 'content-type: application/json' \
        --url "$API?api_key=$RUNPOD_APIKEY" --data "$1")
    if jq -e '.errors' <<<"$resp" >/dev/null 2>&1; then
        jq -r '.errors[].message' <<<"$resp" >&2
        exit 1
    fi
    printf '%s' "$resp"
}

gpu_id() {
    case "${1,,}" in
        4090) echo "NVIDIA GeForce RTX 4090" ;;
        5090) echo "NVIDIA GeForce RTX 5090" ;;
        3090) echo "NVIDIA GeForce RTX 3090" ;;
        a100) echo "NVIDIA A100-SXM4-80GB" ;;
        h100) echo "NVIDIA H100 80GB HBM3" ;;
        l40s) echo "NVIDIA L40S" ;;
        *) echo "$1" ;;
    esac
}

cmd=${1:-help}
case "$cmd" in
launch)
    RUNPOD_APIKEY=$(env_lookup RUNPOD_APIKEY)
    TS_AUTHKEY=$(env_lookup TS_AUTHKEY)
    gpu=$(gpu_id "${2:-4090}")
    netvol=${NETVOL-ucc8x95ndn}
    if [[ -n "$netvol" ]]; then
        cloud=${CLOUD:-SECURE}   # network volumes are secure-cloud only
    else
        cloud=${CLOUD:-COMMUNITY}
    fi
    run_id=${RUN_ID:-main}
    ts_hostname=${TS_HOSTNAME:-snek-train-runpod}
    image=${IMAGE:-ghcr.io/brensch/snek3-trainer:latest}
    payload=$(jq -n \
        --arg gpu "$gpu" --arg cloud "$cloud" --arg image "$image" \
        --arg name "snek3-$ts_hostname" --arg ts "$TS_AUTHKEY" \
        --arg host "$ts_hostname" --arg run "$run_id" --arg start "${START:-}" \
        --arg netvol "$netvol" \
        --argjson vol "${VOLUME_GB:-50}" --argjson disk "${DISK_GB:-20}" '
        {query: "mutation($in: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $in) { id imageName costPerHr machine { gpuDisplayName dataCenterId } } }",
         variables: {in: ({
            cloudType: $cloud, gpuTypeId: $gpu, gpuCount: 1,
            name: $name, imageName: $image,
            volumeMountPath: "/runs", containerDiskInGb: $disk,
            env: [
              {key: "TS_AUTHKEY",  value: $ts},
              {key: "TS_HOSTNAME", value: $host},
              {key: "RUN_ID",      value: $run},
              {key: "START",       value: $start}
            ]} + (if $netvol != "" then {networkVolumeId: $netvol}
                  else {volumeInGb: $vol} end))}}')
    resp=$(gql "$payload")
    jq -r '.data.podFindAndDeployOnDemand
        | "pod \(.id) on \(.machine.gpuDisplayName) at $\(.costPerHr)/hr"' <<<"$resp"
    echo "dashboard (once the tunnel is up): https://$ts_hostname.<tailnet>.ts.net"
    echo "seed it with a run: deploy/runpod.sh sync $run_id $ts_hostname"
    ;;
status)
    RUNPOD_APIKEY=$(env_lookup RUNPOD_APIKEY)
    resp=$(gql '{"query": "query { myself { pods { id name desiredStatus costPerHr machine { gpuDisplayName } runtime { uptimeInSeconds } } } }"}')
    jq -r '.data.myself.pods[]?
        | "\(.id)  \(.name)  \(.machine.gpuDisplayName)  \(.desiredStatus)  $\(.costPerHr)/hr  up \((.runtime.uptimeInSeconds // 0) / 3600 * 10 | round / 10)h"' <<<"$resp"
    ;;
stop)
    RUNPOD_APIKEY=$(env_lookup RUNPOD_APIKEY)
    pod=${2:?usage: deploy/runpod.sh stop <pod-id>}
    gql "$(jq -n --arg id "$pod" \
        '{query: "mutation($in: PodTerminateInput!) { podTerminate(input: $in) }", variables: {in: {podId: $id}}}')" >/dev/null
    echo "terminated $pod"
    ;;
sync)
    run_id=${2:?usage: deploy/runpod.sh sync <run-id> [host]}
    host=${3:-${TS_HOSTNAME:-snek-train-runpod}}
    [[ -d "$REPO_DIR/runs/$run_id" ]] || { echo "error: no runs/$run_id locally" >&2; exit 1; }
    # Tailscale SSH: no keys to manage; the first connection may print a
    # check-mode URL to approve in the browser. tar keeps the pod dependency
    # surface at zero (the image has no rsync).
    echo "pushing runs/$run_id -> root@$host:/runs/$run_id ..."
    tar -C "$REPO_DIR/runs" -czf - "$run_id" | ssh "root@$host" 'tar -xzf - -C /runs'
    echo "done; restart training with RUN_ID=$run_id to resume from it"
    ;;
pull)
    run_id=${2:?usage: deploy/runpod.sh pull <run-id> [host]}
    host=${3:-${TS_HOSTNAME:-snek-train-runpod}}
    echo "pulling root@$host:/runs/$run_id -> runs/$run_id ..."
    ssh "root@$host" "tar -czf - -C /runs '$run_id'" | tar -xzf - -C "$REPO_DIR/runs"
    echo "done"
    ;;
*)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
