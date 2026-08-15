#!/usr/bin/env bash
#
# gpu-xid-watch.sh — fire gpu-crash-collect.sh the moment a severe Xid is logged.
#
# This is the "immediate" half of the crash-evidence capture. The backstop half
# lives in safe-shutdown.sh, which collects before the reboot that would destroy
# the dump. This half exists to cover what the backstop cannot:
#   - an unattended auto-reboot, which never runs safe-shutdown.sh
#   - a fault you do not notice for hours, where collecting immediately captures
#     driver state that is still fresh
#
# systemd has no native "run on kernel log pattern" trigger, so this follows the
# journal itself. Run by gpu-xid-watch.service; not normally invoked by hand.
#
set -uo pipefail

COLLECT=/home/despara/Development/gpu-crash-collect.sh

# Must stay in sync with SEVERE_XIDS in gpu-crash-collect.sh. 13 and 31 are
# excluded on purpose — routine application faults, not hardware events.
SEVERE_RE='NVRM: Xid \(PCI:[^)]*\): (48|63|64|74|79|92|93|94|95|154)[,.]'

# After a trigger the log is still being flooded (the 2026-08-14 fault emitted
# ~250k NVRM lines in six seconds). The collector's latch makes repeat calls
# no-ops, but without a cooldown this loop would spin through every one of those
# lines calling it. Back off instead.
COOLDOWN=60

log() { printf '%s  gpu-xid-watch: %s\n' "$(date '+%H:%M:%S')" "$*"; }

# Catch-up: if this boot ALREADY logged a severe Xid before the watcher started
# (service restarted, or enabled after the fact), collect now. The collector's
# latch means this is a no-op if it was already handled.
if "$COLLECT" --check 2>/dev/null; then
  log "This boot already logged a severe Xid — collecting now."
  "$COLLECT" --full
fi

log "Watching the kernel log for severe Xid faults."
while true; do
  # Blocks until the first matching line. `-n 0` = new entries only; `-m1` makes
  # grep exit on the match, which closes the pipe and ends journalctl too.
  line=$(journalctl -k -f -n 0 -o cat 2>/dev/null | grep -E -m1 "$SEVERE_RE")

  if [ -n "$line" ]; then
    log "SEVERE XID: $line"
    "$COLLECT" --full
    log "Cooling down ${COOLDOWN}s before resuming watch."
    sleep "$COOLDOWN"
  else
    # journalctl exited without a match (journald restart, etc.) — re-establish.
    log "Journal follow ended without a match; re-establishing in 5s."
    sleep 5
  fi
done
