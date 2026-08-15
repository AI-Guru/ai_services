#!/usr/bin/env bash
#
# gpu-aspm-toggle.sh — disable PCIe ASPM on the GPU link, the way that actually
# works on this platform.
#
# BACKGROUND (2026-08-15, learned the hard way):
#
#   `pcie_aspm=off` DOES NOT WORK HERE, and is actively harmful. It does not mean
#   "turn ASPM off on the links" — it means "disable Linux's ASPM driver". With
#   that driver disabled the kernel declines PCIe control at _OSC negotiation:
#
#     with pcie_aspm=off : _OSC: not requesting OS control; OS requires [... ASPM ...]
#     without it         : _OSC: OS now controls [PCIeHotplug SHPCHotplug PME
#                                AER PCIeCapability LTR DPC]
#
#   So firmware keeps ownership, the BIOS-configured `ASPM L1 Enabled` survives
#   untouched, AND the OS loses AER/DPC/hotplug/PME control. Losing AER matters:
#   the PCIe correctable-error counters are the primary diagnostic for the
#   Xid 79 link hypothesis, and firmware-owned AER can leave them reading zero
#   forever while looking perfectly healthy.
#
#   The correct lever is the ASPM *policy*, which keeps the driver (and therefore
#   OS control) and disables L0s/L1 on the links:
#
#     pcie_aspm.policy=performance
#
#   Note the dot: `pcie_aspm.policy=` is a module parameter, `pcie_aspm=` is the
#   driver on/off switch. They are entirely different things.
#
# This remains a MITIGATION ON MECHANISM FIT, not a proven fix — the 2026-08-15
# soak found no errors at all and watched the link retrain 22 times successfully.
#
# Does NOT reboot. Reboot with:
#     sudo /home/despara/Development/safe-shutdown.sh --reboot
#
# Usage:
#   sudo ./gpu-aspm-toggle.sh            # stage pcie_aspm.policy=performance
#                                        # (and strip any pcie_aspm=off)
#   sudo ./gpu-aspm-toggle.sh --runtime  # set the policy NOW, no reboot needed
#   sudo ./gpu-aspm-toggle.sh --revert   # remove both params
#   sudo ./gpu-aspm-toggle.sh --verify   # did it take?
#
set -uo pipefail

ADD_PARAM="pcie_aspm.policy=performance"
DROP_PARAMS="pcie_aspm=off"          # space-separated; always stripped
GRUB=/etc/default/grub
GRUB_CFG=/boot/grub/grub.cfg
POLICY=/sys/module/pcie_aspm/parameters/policy
BDF=0000:01:00.0

MODE=apply
case "${1:-}" in
  --revert)  MODE=revert ;;
  --verify)  MODE=verify ;;
  --runtime) MODE=runtime ;;
  -h|--help) sed -n '2,44p' "$0"; exit 0 ;;
  "") ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;;
esac

ok()   { printf '  \033[32m[ ok ]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m[warn]\033[0m %s\n' "$*"; }
hdr()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

link_ctl() { lspci -vv -s "$BDF" 2>/dev/null | grep -E 'LnkCtl:' | head -1 | sed 's/^\s*//'; }

check_osc() {
  # The precondition for any of this working: the OS must own PCIeCapability.
  local line
  line=$(journalctl -k -b --no-pager 2>/dev/null | grep -E '_OSC: (OS now controls|not requesting)' | tail -1)
  echo "  ${line#*kernel: }"
  case "$line" in
    *"not requesting OS control"*)
      bad "firmware owns PCIe — ASPM policy cannot be applied, and AER is firmware-owned"
      echo "         Almost certainly pcie_aspm=off is still on the cmdline. Run --revert." ;;
    *"OS now controls"*)
      case "$line" in
        *PCIeCapability*) ok "OS controls PCIeCapability — policy changes will apply" ;;
        *) warn "OS has partial control but not PCIeCapability" ;;
      esac ;;
    *) warn "no _OSC line found in this boot's log" ;;
  esac
}

# --------------------------------------------------------------- runtime ----
if [ "$MODE" = runtime ]; then
  [ "$(id -u)" -eq 0 ] || { echo "Re-running with sudo..."; exec sudo "$0" "$@"; }
  hdr "Preconditions"
  check_osc
  hdr "Before"
  echo "  policy:  $(cat "$POLICY" 2>/dev/null)"
  echo "  $(link_ctl)"
  hdr "Setting policy to performance"
  if echo performance >"$POLICY" 2>/tmp/aspm.err; then
    ok "policy written"
  else
    bad "write failed: $(cat /tmp/aspm.err)"
    echo "         This fails when the OS does not control PCIe — see above."
    exit 1
  fi
  hdr "After"
  echo "  policy:  $(cat "$POLICY" 2>/dev/null)"
  lnk=$(link_ctl); echo "  $lnk"
  case "$lnk" in
    *"ASPM Disabled"*) ok "ASPM is now disabled on the link" ;;
    *"ASPM L"*)        bad "link still reports ASPM enabled" ;;
  esac
  echo
  echo "  Runtime only — reverts on reboot. Persist it with: sudo $0"
  exit 0
fi

# ---------------------------------------------------------------- verify ----
if [ "$MODE" = verify ]; then
  hdr "Kernel command line"
  echo "  $(cat /proc/cmdline)"
  grep -qw "$ADD_PARAM" /proc/cmdline \
    && ok "$ADD_PARAM is active" \
    || warn "$ADD_PARAM not on the running kernel"
  for p in $DROP_PARAMS; do
    grep -qw "$p" /proc/cmdline && bad "$p is STILL present — this breaks _OSC; run --revert"
  done

  hdr "PCIe _OSC ownership"
  check_osc

  hdr "ASPM policy"
  echo "  $(cat "$POLICY" 2>/dev/null)"

  hdr "ASPM state on the GPU link"
  if [ "$(id -u)" -ne 0 ]; then
    warn "not root — re-run with sudo to read link capabilities"
  else
    lnk=$(link_ctl); echo "  $lnk"
    case "$lnk" in
      *"ASPM Disabled"*) ok "ASPM is disabled on the link" ;;
      *"ASPM L"*)        bad "ASPM still enabled" ;;
      *)                 warn "could not determine ASPM state" ;;
    esac
  fi

  hdr "Idle link speed"
  echo "  $(cat /sys/bus/pci/devices/$BDF/current_link_speed 2>/dev/null)"
  echo
  echo "  Confirm properly with the soak; the 2026-08-15 ASPM-on baseline was"
  echo "  {2.5: 67, 5.0: 3, 32.0: 287} with 22 transitions. Expect {32.0: N}:"
  echo "    python3 /home/despara/Development/gpu-loadtest.py --duration 1800 --label after-aspm-off"
  exit 0
fi

# ------------------------------------------------------------ apply/revert --
[ "$(id -u)" -eq 0 ] || { echo "Re-running with sudo..."; exec sudo "$0" "$@"; }
[ -f "$GRUB" ] || { bad "$GRUB not found"; exit 1; }

hdr "Current state"
cur_line=$(grep -E '^GRUB_CMDLINE_LINUX_DEFAULT=' "$GRUB" | head -1)
[ -n "$cur_line" ] || { bad "no GRUB_CMDLINE_LINUX_DEFAULT in $GRUB — refusing to guess"; exit 1; }
echo "  $cur_line"
cur_val=$(sed -nE 's/^GRUB_CMDLINE_LINUX_DEFAULT="(.*)"\s*$/\1/p' "$GRUB" | head -1)

# Rebuild as a word list: strip ADD_PARAM and every DROP_PARAM, then re-add
# ADD_PARAM only when applying. Idempotent in both directions.
new_val=""
for w in $cur_val; do
  [ "$w" = "$ADD_PARAM" ] && continue
  skip=0
  for d in $DROP_PARAMS; do [ "$w" = "$d" ] && skip=1; done
  [ "$skip" -eq 1 ] && continue
  new_val="${new_val:+$new_val }$w"
done
[ "$MODE" = apply ] && new_val="${new_val:+$new_val }$ADD_PARAM"

if [ "$new_val" = "$cur_val" ]; then
  ok "already in the desired state (mode=$MODE): \"$cur_val\""
  exit 0
fi

hdr "Backup"
BAK="${GRUB}.bak-$(date +%Y%m%d-%H%M%S)"
cp -a "$GRUB" "$BAK" && ok "saved $BAK" || { bad "backup failed — aborting"; exit 1; }

hdr "Change"
sed -i -E "s|^GRUB_CMDLINE_LINUX_DEFAULT=\".*\"\s*$|GRUB_CMDLINE_LINUX_DEFAULT=\"$new_val\"|" "$GRUB"
if diff -u "$BAK" "$GRUB"; then
  bad "file unchanged — sed did not match; restore from $BAK"
  exit 1
fi

hdr "Regenerate grub.cfg"
if update-grub >/tmp/update-grub.log 2>&1 || grub-mkconfig -o "$GRUB_CFG" >/tmp/update-grub.log 2>&1; then
  ok "grub config regenerated"
else
  bad "grub regeneration FAILED — restoring from backup"
  cp -a "$BAK" "$GRUB"; tail -20 /tmp/update-grub.log | sed 's/^/      /'; exit 1
fi

hdr "Verify generated config"
# NOT `$(grep -c ... || echo 0)`. On zero matches grep PRINTS "0" and THEN exits
# 1, so the `|| echo 0` appends a second line and the variable becomes "0\n0" —
# which blows up the arithmetic below with "syntax error in expression". grep -c
# already prints 0 when it finds nothing, and this script does not use `set -e`,
# so the non-zero exit is harmless and needs no fallback.
hits=$(grep -c -- "$ADD_PARAM" "$GRUB_CFG" 2>/dev/null)
stale=0
for d in $DROP_PARAMS; do
  n=$(grep -c -- "$d" "$GRUB_CFG" 2>/dev/null)
  stale=$((stale + ${n:-0}))
done
[ "$stale" -eq 0 ] && ok "no stale $DROP_PARAMS in $GRUB_CFG" || bad "$stale stale entries remain"
hits=${hits:-0}   # empty if grub.cfg was unreadable; never leave it unset
if [ "$MODE" = apply ]; then
  [ "$hits" -gt 0 ] && ok "$ADD_PARAM present ($hits boot entries)" || bad "$ADD_PARAM NOT in $GRUB_CFG"
else
  [ "$hits" -eq 0 ] && ok "$ADD_PARAM removed" || bad "$ADD_PARAM still present ($hits)"
fi

hdr "Next"
cat <<EOF
  Staged, NOT active until reboot.

      sudo /home/despara/Development/safe-shutdown.sh --reboot
      sudo $0 --verify

  You can also test the policy immediately after that reboot without a further
  one — but only once pcie_aspm=off is gone and the OS owns PCIe again:

      sudo $0 --runtime

  Roll back:  sudo $0 --revert   (or: cp -a $BAK $GRUB && update-grub)
EOF
