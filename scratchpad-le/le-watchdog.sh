#!/usr/bin/env bash
# Overnight watchdog for the snek3-le-1 run. Deterministic auto-recovery:
#  - server process gone  -> relaunch binary, wait for port, resume run
#  - server up, not running-> resume run
#  - generation stalled    -> log (Claude cron decides on restart)
# Smart alerting (entropy collapse, quality) is left to the hourly Claude cron;
# this just keeps the process alive. Logs every action + a heartbeat.
set -uo pipefail
cd /home/brensch/snek3
LOG=/home/brensch/snek3/scratchpad-le/watchdog.log
RUN=snek3-le-6
API=http://127.0.0.1:8050

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

port_up()   { ss -ltn 2>/dev/null | grep -q ':8050 '; }
state()     { curl -s --max-time 5 "$API/api/state" 2>/dev/null; }

launch_server() {
  log "server down -> launching"
  setsid bash /home/brensch/snek3/scratchpad-le/le-server.sh \
    >> /home/brensch/snek3/scratchpad-le/server.log 2>&1 &
  for _ in $(seq 1 60); do port_up && break; sleep 1; done
  port_up && log "server up" || log "server FAILED to come up in 60s"
}

resume_run() {
  log "run not active -> POST start (resume)"
  curl -s --max-time 10 -X POST "$API/api/control/start" \
    -H 'content-type: application/json' \
    -d "{\"run_id\":\"$RUN\",\"fresh\":false}" >> "$LOG" 2>&1
  echo >> "$LOG"
}

last_gen=-1
last_gen_change=$(date +%s)
log "watchdog started (pid $$)"
while true; do
  if ! port_up; then
    # A dead PROCESS is the only thing we auto-revive: a graceful "running:
    # false" can only come from the API (a deliberate stop by the user or the
    # agent) and must stick. Crashes kill the whole server, so relaunch+resume
    # here covers every real failure without overriding intentional stops.
    launch_server
    sleep 20
    resume_run
    last_gen=-1; last_gen_change=$(date +%s)
    sleep 120; continue
  fi
  s=$(state)
  running=$(echo "$s" | grep -o '"running":[a-z]*' | cut -d: -f2)
  gen=$(echo "$s" | grep -o '"generation":[0-9]*' | cut -d: -f2)
  now=$(date +%s)
  if [ "$running" != "true" ]; then
    # Intentional stop (see above) — just note it and keep watching.
    if [ $((now % 600)) -lt 120 ]; then log "run stopped (intentional) — not resuming"; fi
    last_gen=-1; last_gen_change=$now
    sleep 120; continue
  fi
  if [ -n "${gen:-}" ] && [ "$gen" != "$last_gen" ]; then
    last_gen=$gen; last_gen_change=$now
  elif [ $((now - last_gen_change)) -gt 480 ]; then
    log "STALL: gen stuck at $last_gen for >8min while running=true"
    last_gen_change=$now   # don't spam; re-flag every 8min
  fi
  # heartbeat every ~10 min
  if [ $((now % 600)) -lt 120 ]; then log "heartbeat gen=$gen running=$running"; fi
  sleep 120
done
