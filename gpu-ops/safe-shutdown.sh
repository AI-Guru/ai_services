#!/usr/bin/env bash
#
# safe-shutdown.sh — release the GPU cleanly, then power off.
#
# Why this exists:
#   `sudo shutdown now` lets systemd stop docker.service, which force-kills any
#   running inference container after only ~10s. Killing vLLM mid-CUDA-op wedges
#   the RTX PRO 6000 into a reset-required state (Xid 154). This Blackwell card
#   has NO `nvidia-smi --gpu-reset` support ("Not Supported"), and a warm reboot
#   does NOT clear it — the only recovery is a cold power cycle (drop mains).
#
#   This script stops every GPU-holding container gracefully first (long SIGTERM
#   grace so vLLM can release CUDA), confirms the GPU is idle, and only then
#   triggers the shutdown. Use it INSTEAD of `sudo shutdown now`.
#
# Usage:
#   sudo /home/despara/Development/safe-shutdown.sh               # graceful GPU release, then power off
#   sudo /home/despara/Development/safe-shutdown.sh --reboot      # ... reboot instead of power off
#        /home/despara/Development/safe-shutdown.sh --dry-run     # stop GPU containers + verify, but DON'T shut down
#        /home/despara/Development/safe-shutdown.sh --release-only # stop GPU containers + verify only (used by the systemd unit)
#   sudo /home/despara/Development/safe-shutdown.sh -t 120        # override the per-container stop timeout (default 90s)
#
set -uo pipefail

STOP_TIMEOUT=90          # seconds vLLM gets to release CUDA before docker SIGKILLs it
ACTION=poweroff
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|--release-only) DRY_RUN=1 ;;
    --reboot)   ACTION=reboot ;;
    --poweroff) ACTION=poweroff ;;
    -t)         STOP_TIMEOUT="${2:?-t needs a value}"; shift ;;
    -h|--help)  sed -n '2,25p' "$0"; exit 0 ;;
    *)          echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

log() { printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }

# 1. If this boot hit a severe Xid, persist the GPU crash dump BEFORE we tear
#    anything down. The driver keeps that dump only until the kernel module is
#    unloaded, so the reboot we are about to perform is what destroys it — this
#    is the last chance to capture it. Collecting first (rather than after the
#    containers are stopped) also preserves the process/memory state that was
#    live at the time of the fault.
#
#    Costs nothing on a normal shutdown: with no severe Xid in the log the
#    collector exits in well under a second. gpu-xid-watch.service usually gets
#    there first; the collector's latch makes this a no-op in that case.
COLLECT=/home/despara/Development/gpu-crash-collect.sh
if [ -x "$COLLECT" ]; then
  if [ "$(id -u)" -eq 0 ]; then
    "$COLLECT" || log "WARNING: crash collection failed (continuing with shutdown)."
  elif "$COLLECT" --check 2>/dev/null; then
    log "WARNING: severe Xid logged this boot but not running as root —"
    log "         crash dump NOT collected. Run: sudo $COLLECT --full"
  fi
fi

# 2. Find every container that requested GPU devices (survives whichever model
#    is loaded). We enumerate `docker ps -a` (NOT just running) on purpose: a
#    crash-looping container flickers through an `exited` state between restarts,
#    so `docker ps` alone could miss it — then its restart policy would bounce it
#    back up and docker.service's own 10s-SIGKILL would wedge the GPU. Listing
#    it here and `docker stop`ping it by name cancels the restart loop for good,
#    whatever phase of the loop it's in. Stopping an already-exited container is
#    a harmless no-op.
mapfile -t GPU_CONTAINERS < <(
  docker ps -a --format '{{.Names}} {{.State}}' | while read -r n state; do
    [ "$state" = dead ] && continue
    [ -n "$(docker inspect -f '{{if .HostConfig.DeviceRequests}}1{{end}}' "$n" 2>/dev/null)" ] && echo "$n"
  done
)

if [ "${#GPU_CONTAINERS[@]}" -eq 0 ]; then
  log "No GPU containers present — nothing to release."
else
  log "Stopping GPU containers gracefully (timeout ${STOP_TIMEOUT}s each): ${GPU_CONTAINERS[*]}"
  # `docker stop` sends SIGTERM, waits up to the timeout, then SIGKILL, AND
  # cancels the container's restart policy for this stop — so a crash-looper
  # can't bounce back. vLLM releases the GPU cleanly on SIGTERM if given enough
  # time; the 10s default is what wedged the card, so we use a generous timeout.
  docker stop -t "$STOP_TIMEOUT" "${GPU_CONTAINERS[@]}"
fi

# 3. Verify the GPU has drained (no compute processes) before continuing.
procs=1
for _ in $(seq 1 12); do
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)
  memused=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${procs:-0}" -eq 0 ] && { log "GPU idle — no compute processes (${memused:-?} MiB used)."; break; }
  log "Waiting for GPU to drain: ${procs} process(es), ${memused:-?} MiB used..."
  sleep 5
done

if [ "${procs:-1}" -ne 0 ]; then
  log "WARNING: GPU still has ${procs} compute process(es) after draining."
  log "ABORTING shutdown to avoid wedging the card. Investigate: nvidia-smi ; docker ps"
  exit 1
fi

# 4. Shut down (or stop here in dry-run / --release-only).
if [ "$DRY_RUN" -eq 1 ]; then
  log "GPU released cleanly. NOT shutting down (release-only mode)."
  exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
  log "Need root to ${ACTION}. Re-run with sudo, or run: sudo shutdown -${ACTION:0:1} now"
  # -h for poweroff, -r for reboot; handled explicitly below when root.
  exit 1
fi

log "GPU released cleanly. Triggering ${ACTION}..."
if [ "$ACTION" = reboot ]; then
  shutdown -r now
else
  shutdown -h now
fi
