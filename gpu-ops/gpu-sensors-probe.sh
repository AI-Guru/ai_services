#!/usr/bin/env bash
#
# gpu-sensors-probe.sh — try to expose motherboard voltage rails (+12V in
# particular), which is the only software route to watching for PSU sag.
#
# Why: the ASUS TUF GAMING B850-PLUS WIFI exposes an `asus` hwmon device that is
# a stub — no sensor inputs at all. The only voltages readable on this box come
# from the integrated amdgpu. The board almost certainly carries a Nuvoton
# super-I/O chip that the nct6775 driver can drive; it just is not loaded.
#
# If this works, +12V rail voltage becomes visible and a sag under GPU load
# becomes a measurable event rather than a hypothesis.
#
# Usage: sudo /home/despara/Development/gpu-sensors-probe.sh
#
set -uo pipefail

[ "$(id -u)" -eq 0 ] || { echo "Re-running with sudo..."; exec sudo "$0" "$@"; }

log() { printf '  %s\n' "$*"; }
hdr() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# NOT `lsmod | grep -q ...`. Under `set -o pipefail`, grep -q exits the moment it
# matches, lsmod then takes SIGPIPE writing the rest of its output, and pipefail
# surfaces that 141 as pipeline failure — so a LOADED module reports as absent.
# It is timing-dependent (it does not reproduce on every run), which is exactly
# what made the first version of this script claim "nct6775 not loaded"
# immediately after printing "modprobe nct6775: OK". The here-string reads
# lsmod's output in full first, so there is no pipe to break.
nct_loaded() { grep -q '^nct6775' <<<"$(lsmod)"; }

hdr "Before"
log "hwmon devices currently present:"
for h in /sys/class/hwmon/hwmon*; do
  printf '    %-10s %s\n' "$(basename "$h")" "$(cat "$h/name" 2>/dev/null)"
done

hdr "Loading nct6775"
if nct_loaded; then
  log "already loaded."
else
  # Two attempts. The plain load usually fails on modern ASUS boards because the
  # ACPI tables claim the super-I/O IO ports, and Linux refuses to touch them
  # unless told the conflict is acceptable. That override is a BOOT parameter —
  # it cannot be set at modprobe time, which is why the fallback below only
  # prints instructions rather than retrying.
  if modprobe nct6775 2>/tmp/nct.err; then
    log "modprobe nct6775: OK"
  else
    log "modprobe nct6775 FAILED:"
    sed 's/^/      /' /tmp/nct.err
    if dmesg 2>/dev/null | tail -40 | grep -qi 'ACPI.*conflict\|resource.*conflict'; then
      log ""
      log "  Cause: ACPI resource conflict — expected on this board."
      log "  Fix requires a boot parameter: acpi_enforce_resources=lax"
      log "  Add it in the same reboot as pcie_aspm=off so you only reboot once."
    fi
  fi
fi

hdr "After — any voltage/current rails now?"
found=0
for f in /sys/class/hwmon/hwmon*/in*_input /sys/class/hwmon/hwmon*/curr*_input; do
  [ -f "$f" ] || continue
  n=$(cat "$(dirname "$f")/name" 2>/dev/null)
  lbl_file="${f%_input}_label"
  lbl=$( [ -f "$lbl_file" ] && cat "$lbl_file" || basename "$f" )
  raw=$(cat "$f" 2>/dev/null)
  # hwmon reports millivolts / milliamps.
  printf '    %-10s %-22s %s (%.3f)\n' "$n" "$lbl" "$raw" "$(echo "$raw" | awk '{print $1/1000}')"
  found=1
done
[ "$found" -eq 1 ] || log "    none — no rail telemetry available (amdgpu-only readings are not PSU rails)"

hdr "Result"
if nct_loaded; then
  echo "  nct6775 loaded (chip: NCT6799). Rails are exported to Prometheus as"
  echo "  board_voltage_raw_volts by gpu-events-exporter."
  echo
  echo "  NOTE: the driver labels temperatures but NOT voltages — each rail sits"
  echo "  behind a board-specific divider, so values are RAW. in16 (~1.84V) is the"
  echo "  likely +12V at roughly 6.6:1. Confirm empirically: under GPU load the"
  echo "  +12V rail is the one that droops as power draw climbs."
  echo
  # Persist across reboots, idempotently.
  CONF=/etc/modules-load.d/nct6775.conf
  if [ -f "$CONF" ] && grep -q '^nct6775$' "$CONF"; then
    echo "  Persistence: already configured ($CONF)."
  else
    echo nct6775 >"$CONF" && echo "  Persistence: wrote $CONF (loads at every boot)."
  fi
else
  echo "  nct6775 not loaded — see the boot-parameter note above."
  echo "  This is not a blocker: the HW Power Braking counter is already being"
  echo "  scraped and is the more direct probe for a power-delivery fault."
fi
