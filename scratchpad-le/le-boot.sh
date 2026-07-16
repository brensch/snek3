#!/usr/bin/env bash
# Boot-recovery for the snek3 training stack: bring up the server, resume the
# active run, and start the watchdog. Idempotent — safe to run when already up.
# Wired to a Windows logon scheduled task so a host reboot (update, power cut)
# self-heals instead of silently killing training.
#
# Respects an intentional stop: if the newest experiments.log control marker is
# INTENTIONAL STOP, do nothing.
set -uo pipefail
cd /home/brensch/snek3
LOG=scratchpad-le/boot.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

RUN=snek3-le-6

# Most recent control marker wins: a later resume-style line overrides a stop.
last_stop=$(grep -n 'INTENTIONAL STOP' scratchpad-le/experiments.log | tail -1 | cut -d: -f1)
last_resume=$(grep -n 'service resumed\|REGIME.*relaunch\|RELAUNCH' scratchpad-le/experiments.log | tail -1 | cut -d: -f1)
if [ -n "${last_stop:-}" ] && [ "${last_stop:-0}" -gt "${last_resume:-0}" ]; then
  log "intentional stop is the latest control marker — not starting"
  exit 0
fi

if ! ss -ltn 2>/dev/null | grep -q ':8050 '; then
  log "server down — launching"
  setsid bash scratchpad-le/le-server.sh >> scratchpad-le/server.log 2>&1 &
  for _ in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q ':8050 ' && break; sleep 1; done
fi
sleep 3
running=$(curl -s --max-time 5 http://127.0.0.1:8050/api/state | grep -o '"running":[a-z]*' | cut -d: -f2)
if [ "$running" != "true" ]; then
  log "resuming $RUN"
  curl -s --max-time 10 -X POST http://127.0.0.1:8050/api/control/start \
    -H 'content-type: application/json' \
    -d "{\"run_id\":\"$RUN\",\"fresh\":false}" >> "$LOG" 2>&1
  echo >> "$LOG"
fi
if ! pgrep -f 'l[e]-watchdog' >/dev/null; then
  log "watchdog down — launching"
  setsid bash scratchpad-le/le-watchdog.sh >/dev/null 2>&1 &
fi
log "boot recovery done"
