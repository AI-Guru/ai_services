#!/usr/bin/env bash
#
# gpu-power-cap.sh — set and persist the GPU board power limit.
#
# WHY THIS EXISTS (2026-09-05):
#   Two Xid 79 bus drops, and the second one happened at 598 W sustained /
#   603 W peak against a 600 W cap, while ~1 h of soak testing at <=475 W has
#   never faulted. Then a physical inspection found white flaking at the
#   12V-2x6 connector's retention latch.
#
#   If that flaking is stress-whitened/cracked plastic, the latch may not be
#   holding the connector fully mated. Partial seating on a 12V-2x6 carrying
#   ~50 A is the documented root cause of connector melting, and would explain
#   an instantaneous power interruption that software cannot see: no AER, no
#   thermal event, no precursor.
#
#   Capping the board power reduces peak current through a possibly compromised
#   connector. This is a SAFETY MEASURE while the connector is assessed, not a
#   fix. Do not treat a quiet period at a reduced cap as proof the hardware is
#   healthy.
#
# Usage:
#   sudo ./gpu-power-cap.sh 500       # apply now + persist across reboots
#   sudo ./gpu-power-cap.sh --show    # current limit
#   sudo ./gpu-power-cap.sh --remove  # back to the 600 W default, drop persistence
#
set -uo pipefail

UNIT=/etc/systemd/system/gpu-power-cap.service
DEFAULT_W=600

ok()   { printf '  \033[32m[ ok ]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; }
hdr()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

show() {
  nvidia-smi --query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit \
             --format=csv 2>/dev/null | sed 's/^/  /'
  if systemctl is-enabled --quiet gpu-power-cap.service 2>/dev/null; then
    ok "persistence enabled: $(grep -oP 'ExecStart=.*-pl \K[0-9]+' "$UNIT" 2>/dev/null) W at boot"
  else
    echo "  (no boot-time persistence configured)"
  fi
}

case "${1:-}" in
  --show) hdr "Current"; show; exit 0 ;;
  -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
esac

[ "$(id -u)" -eq 0 ] || { echo "Re-running with sudo..."; exec sudo "$0" "$@"; }

if [ "${1:-}" = "--remove" ]; then
  hdr "Removing cap"
  nvidia-smi -pl "$DEFAULT_W" >/dev/null 2>&1 && ok "limit back to ${DEFAULT_W} W" || bad "could not reset limit"
  systemctl disable --now gpu-power-cap.service >/dev/null 2>&1 && ok "persistence disabled"
  rm -f "$UNIT"; systemctl daemon-reload
  hdr "Now"; show; exit 0
fi

W="${1:?usage: $0 <watts> | --show | --remove}"
case "$W" in ''|*[!0-9]*) bad "watts must be an integer"; exit 2 ;; esac
MIN=$(nvidia-smi --query-gpu=power.min_limit --format=csv,noheader,nounits | cut -d. -f1)
MAX=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | cut -d. -f1)
if [ "$W" -lt "${MIN:-150}" ] || [ "$W" -gt "${MAX:-600}" ]; then
  bad "$W W is outside the card's ${MIN}-${MAX} W range"; exit 2
fi

hdr "Before"; show

hdr "Applying ${W} W"
if nvidia-smi -pl "$W" 2>&1 | sed 's/^/  /'; then ok "applied"; else bad "apply failed"; exit 1; fi

hdr "Persisting"
# nvidia-smi -pl does NOT survive a reboot, and persistence mode does not carry
# it either -- it needs re-applying at every boot.
cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=Cap the GPU board power limit (Xid 79 mitigation - see GPU-XID79-LOG.md)
After=nvidia-persistenced.service systemd-modules-load.service
Wants=systemd-modules-load.service

[Service]
Type=oneshot
RemainAfterExit=yes
# Retry: the driver may not be ready the instant this unit is reached at boot.
ExecStart=/bin/sh -c 'for i in 1 2 3 4 5; do /usr/bin/nvidia-smi -pl ${W} && exit 0; sleep 5; done; exit 1'

[Install]
WantedBy=multi-user.target
UNIT_EOF
systemctl daemon-reload
systemctl enable gpu-power-cap.service >/dev/null 2>&1 && ok "enabled at boot (${W} W)" || bad "enable failed"

hdr "After"; show
echo
echo "  Reminder: this reduces peak current through the connector. It is a"
echo "  mitigation while the 12V-2x6 latch is assessed -- NOT a fix, and a quiet"
echo "  period at this cap is NOT evidence the hardware is sound."
