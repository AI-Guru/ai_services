#!/usr/bin/env bash
# Restart GPU exporters that have gone blind.
#
# Why this exists: on 2026-09-03 nvidia-gpu-exporter lost its device handles and
# spent 10.5 hours returning HTTP 200 with no GPU metrics in it. The container
# stayed "Up", the Prometheus target stayed "up", and the day's power history was
# simply gone. Docker's `restart: always` does not help here — restart policies
# ignore health state, so an unhealthy container sits there indefinitely.
#
# Three independent checks, because any one can fail alone:
#   1. Docker health state         — catches NVML dying inside the container
#   2. nvidia-smi exec             — the direct probe, and the one that actually
#                                    caught the 2026-09-03 fault
#   3. failed_scrapes_total > last — secondary net
#
# On (3): the exporter does NOT publish nvidia_smi_failed_scrapes_total at all
# while it is healthy — the series only appears once a scrape has failed (it read
# 7520 during the incident). So the baseline is treated as 0 when the metric is
# absent, and any value above the last seen one counts as a fault. The counter
# resets to 0 on restart, so the stored baseline is dropped after a recovery.
#
# Install (user crontab, no root needed — docker-group membership is enough):
#   */5 * * * * /home/despara/Development/ai_services/gpu-dashboard/exporter-watchdog.sh
#
# Logs to syslog under the tag gpu-exporter-watchdog:
#   journalctl -t gpu-exporter-watchdog --since today
set -uo pipefail

CONTAINER=nvidia-gpu-exporter
METRICS_URL=http://localhost:9835/metrics
STATE=/tmp/.gpu-exporter-watchdog.state
TAG=gpu-exporter-watchdog

log() { logger -t "$TAG" -- "$*"; echo "$(date '+%F %T') $*"; }

docker inspect "$CONTAINER" >/dev/null 2>&1 || { log "container $CONTAINER absent — nothing to do"; exit 0; }

health=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$CONTAINER" 2>/dev/null)
running=$(docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null)

# current cumulative failure counter (empty if unreachable)
now=$(curl -s --max-time 10 "$METRICS_URL" 2>/dev/null \
      | awk '/^nvidia_smi_failed_scrapes_total /{print int($2)}')
prev=$(cat "$STATE" 2>/dev/null || echo "")
[ -n "${now:-}" ] && echo "$now" > "$STATE"

reason=""
if [ "$running" != "true" ]; then
  reason="not running (Running=$running)"
elif [ "$health" = unhealthy ]; then
  reason="healthcheck unhealthy"
elif [ -n "${now:-}" ] && [ "$now" -gt "${prev:-0}" ]; then
  reason="failed_scrapes_total rose ${prev:-0} -> $now since last check"
elif ! docker exec "$CONTAINER" nvidia-smi --query-gpu=power.draw --format=csv,noheader >/dev/null 2>&1; then
  reason="nvidia-smi fails inside the container"
fi

[ -z "$reason" ] && exit 0

log "restarting $CONTAINER: $reason"
if docker restart "$CONTAINER" >/dev/null 2>&1; then
  sleep 12
  if docker exec "$CONTAINER" nvidia-smi --query-gpu=power.draw --format=csv,noheader >/dev/null 2>&1; then
    log "recovered — nvidia-smi works inside the container again"
    # counter resets to 0 on restart; drop the stale baseline
    rm -f "$STATE"
  else
    # A restart that does not fix it means the fault is below Docker (driver
    # reload, Xid, wedged card). Say so instead of restarting in a loop —
    # see GPU-INCIDENT-RUNBOOK.md.
    log "STILL BROKEN after restart — suspect driver/GPU level, see GPU-INCIDENT-RUNBOOK.md"
  fi
else
  log "docker restart failed"
fi
