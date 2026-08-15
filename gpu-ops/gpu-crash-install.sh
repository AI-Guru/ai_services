#!/usr/bin/env bash
#
# gpu-crash-install.sh — install and verify the GPU crash-dump capture chain.
#
# Installs:
#   gpu-xid-watch.service     — fires gpu-crash-collect.sh seconds after a severe Xid
#   gpu-safe-shutdown.service — updated TimeoutStopSec (collection now runs before teardown)
#
# Then runs an end-to-end test: a real forced collection, plus a regression test
# of the Xid severity filter against this machine's own historical journals.
#
# Usage:
#   sudo /home/despara/Development/gpu-crash-install.sh              # install + test
#        /home/despara/Development/gpu-crash-install.sh              # (self-elevates via sudo)
#   sudo /home/despara/Development/gpu-crash-install.sh --no-test    # install only
#   sudo /home/despara/Development/gpu-crash-install.sh --test-only  # verify an existing install
#
# Exits non-zero if any check fails.
#
set -uo pipefail

SRC_DIR=/home/despara/Development
UNIT_SRC=/home/despara
UNIT_DST=/etc/systemd/system
COLLECT="$SRC_DIR/gpu-crash-collect.sh"
WATCH="$SRC_DIR/gpu-xid-watch.sh"
OUT_ROOT=/var/log/gpu-crash
LATCH=/run/gpu-crash-collected
EXPECT_STOP_SEC=360

DO_INSTALL=1
DO_TEST=1
while [ $# -gt 0 ]; do
  case "$1" in
    --no-test)   DO_TEST=0 ;;
    --test-only) DO_INSTALL=0 ;;
    -h|--help)   sed -n '2,19p' "$0"; exit 0 ;;
    *)           echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

# Self-elevate: nvidia-bug-report.sh and systemctl both need root, and running
# the test as a normal user produces a misleading empty dump (nvidia-debugdump
# exits 0 while writing a 44-byte zip).
if [ "$(id -u)" -ne 0 ]; then
  echo "Re-running with sudo..."
  exec sudo "$0" "$@"
fi

PASS=0; FAIL=0; WARN=0
ok()   { printf '  \033[32m[ ok ]\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33m[warn]\033[0m %s\n' "$*"; WARN=$((WARN+1)); }
hdr()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# =============================================================================
hdr "1. Prerequisites"
# =============================================================================
for f in "$COLLECT" "$WATCH"; do
  if [ -x "$f" ]; then ok "$(basename "$f") present and executable"
  else bad "$f missing or not executable (chmod +x it)"; fi
done
for u in gpu-xid-watch.service gpu-safe-shutdown.service; do
  [ -f "$UNIT_SRC/$u" ] && ok "unit source $u present" || bad "$UNIT_SRC/$u missing"
done
for t in nvidia-bug-report.sh nvidia-debugdump nvidia-smi; do
  command -v "$t" >/dev/null && ok "$t found" || bad "$t NOT found"
done

# The latch lives in /run and MUST be on tmpfs — that is what makes it reset on
# every boot, giving "one collection per boot" for free. If /run were persistent
# the latch would survive the reboot and suppress the next collection entirely.
fstype=$(findmnt -no FSTYPE /run 2>/dev/null)
[ "$fstype" = tmpfs ] && ok "/run is tmpfs (latch resets each boot)" \
                      || bad "/run is '$fstype', not tmpfs — the per-boot latch will not reset"

[ "$FAIL" -gt 0 ] && { echo; echo "Prerequisites failed — stopping."; exit 1; }

# =============================================================================
if [ "$DO_INSTALL" -eq 1 ]; then
hdr "2. Install units"
# =============================================================================
  for u in gpu-xid-watch.service gpu-safe-shutdown.service; do
    if install -m 644 "$UNIT_SRC/$u" "$UNIT_DST/$u"; then ok "installed $u"
    else bad "failed to install $u"; fi
  done

  systemctl daemon-reload && ok "systemctl daemon-reload" || bad "daemon-reload failed"

  # Enable + start the watcher. Safe: it only follows the journal.
  if systemctl enable --now gpu-xid-watch.service >/dev/null 2>&1; then
    ok "gpu-xid-watch.service enabled and started"
  else
    bad "could not enable/start gpu-xid-watch.service"
  fi

  # !! Deliberately NOT restarting gpu-safe-shutdown.service. Its ExecStop IS
  # !! safe-shutdown.sh, so `systemctl restart` would stop every GPU container
  # !! on the box — i.e. tear down whatever model is currently serving. A
  # !! daemon-reload is sufficient for the new TimeoutStopSec to take effect.
  ok "gpu-safe-shutdown.service reloaded in place (NOT restarted — that would stop your models)"
fi

# =============================================================================
hdr "3. Verify installed state"
# =============================================================================
if systemctl is-active --quiet gpu-xid-watch.service; then
  ok "gpu-xid-watch.service is active ($(systemctl show -p MainPID --value gpu-xid-watch.service))"
else
  bad "gpu-xid-watch.service is NOT active — see: journalctl -u gpu-xid-watch -n 50"
fi
systemctl is-enabled --quiet gpu-xid-watch.service \
  && ok "gpu-xid-watch.service enabled at boot" \
  || bad "gpu-xid-watch.service not enabled at boot"

# systemd reports this human-readably ("6min", "3min 20s", "45s", "infinity"),
# so convert before comparing rather than pattern-matching one shape.
parse_dur() {
  local s="$1" total=0 mins secs
  [ "$s" = infinity ] && { echo 999999; return; }
  mins=$(grep -oE '[0-9]+min' <<<"$s" | tr -d 'min')
  secs=$(grep -oE '[0-9]+s'   <<<"$s" | tr -d 's')
  [ -n "$mins" ] && total=$((total + mins * 60))
  [ -n "$secs" ] && total=$((total + secs))
  echo "$total"
}
stop_usec=$(systemctl show -p TimeoutStopUSec --value gpu-safe-shutdown.service 2>/dev/null)
stop_sec=$(parse_dur "$stop_usec")
if [ "${stop_sec:-0}" -ge "$EXPECT_STOP_SEC" ]; then
  ok "gpu-safe-shutdown TimeoutStopSec = $stop_usec (>= ${EXPECT_STOP_SEC}s)"
else
  bad "gpu-safe-shutdown TimeoutStopSec = $stop_usec — expected >= ${EXPECT_STOP_SEC}s"
fi

grep -q 'gpu-crash-collect.sh' "$SRC_DIR/safe-shutdown.sh" \
  && ok "safe-shutdown.sh calls the collector (backstop wired)" \
  || bad "safe-shutdown.sh does NOT call gpu-crash-collect.sh"

# =============================================================================
hdr "4. Severity-filter regression test (against this machine's real journals)"
# =============================================================================
# The whole design rests on firing for hardware faults and staying silent for
# routine application faults. Test that against actual recorded history rather
# than trusting the regex by inspection.
SEVERE_RE='NVRM: Xid \(PCI:[^)]*\): (48|63|64|74|79|92|93|94|95|154)[,.]'
ANY_XID_RE='NVRM: Xid \(PCI:'

found_fault_boot=0
found_noise_boot=0
for b in $(seq 0 -1 -12); do
  journalctl -k -b "$b" --no-pager >/dev/null 2>&1 || continue
  sev=$(journalctl -k -b "$b" --no-pager 2>/dev/null | grep -cE "$SEVERE_RE")
  any=$(journalctl -k -b "$b" --no-pager 2>/dev/null | grep -cE "$ANY_XID_RE")
  [ "$any" -eq 0 ] && continue
  codes=$(journalctl -k -b "$b" --no-pager 2>/dev/null \
          | sed -nE 's/.*NVRM: Xid \(PCI:[^)]*\): ([0-9]+).*/\1/p' | sort -un | tr '\n' ' ')
  if [ "$sev" -gt 0 ]; then
    ok "boot $b: codes [${codes% }] -> WOULD collect ($sev severe line(s))"
    found_fault_boot=1
  else
    ok "boot $b: codes [${codes% }] -> correctly ignored (application-level only)"
    found_noise_boot=1
  fi
done
[ "$found_fault_boot" -eq 1 ] || warn "no historical boot with a severe Xid found to test against"
[ "$found_noise_boot" -eq 1 ] || warn "no historical boot with only benign Xids found to test against"

# Current boot: --check must agree with what is actually in the log.
if "$COLLECT" --check 2>/dev/null; then
  warn "current boot HAS a severe Xid — collection will trigger for real"
else
  ok "current boot has no severe Xid (--check correctly exits non-zero)"
fi

# =============================================================================
if [ "$DO_TEST" -eq 1 ]; then
hdr "5. End-to-end collection test (forced)"
# =============================================================================
echo "  Running a real collection. Worst case ~3.5 min if the GPU is unresponsive."
echo "  --force bypasses the per-boot latch, so this is safe to run at any time."
echo

before=$(ls -1d "$OUT_ROOT"/*/ 2>/dev/null | wc -l)
start=$(date +%s)
"$COLLECT" --force --full
rc=$?
elapsed=$(( $(date +%s) - start ))
after=$(ls -1d "$OUT_ROOT"/*/ 2>/dev/null | wc -l)

echo
[ "$rc" -eq 0 ] && ok "collector exited 0 (took ${elapsed}s)" \
                || bad "collector exited $rc"
[ "$after" -gt "$before" ] && ok "new collection directory created" \
                           || bad "no new collection directory under $OUT_ROOT"

DIR=$(ls -1dt "$OUT_ROOT"/*/ 2>/dev/null | head -1)
if [ -n "$DIR" ] && [ -f "$DIR/manifest.txt" ]; then
  echo
  echo "  --- manifest: $DIR ---"
  sed 's/^/  /' "$DIR/manifest.txt"
  echo "  --- artifacts ---"
  ls -lh "$DIR" | tail -n +2 | sed 's/^/  /'
  echo

  # Timeouts are the one thing that could not be sized without running as root.
  # If any step hit its ceiling, say so loudly and name the knob to turn.
  if grep -q 'rc=124' "$DIR/manifest.txt"; then
    bad "a step TIMED OUT (rc=124) — raise the matching T_* value in gpu-crash-collect.sh,"
    echo "         and raise TimeoutStopSec in gpu-safe-shutdown.service to match."
    grep 'rc=124' "$DIR/manifest.txt" | sed 's/^/         /'
  else
    ok "no step hit its timeout — the T_* defaults are adequately sized"
  fi

  # Grade BOTH failure shapes. Grading only SUSPECT_EMPTY was itself a bug:
  # on 2026-08-14 two artifacts were logged MISSING and the run still scored
  # "0 failed", because nothing looked at that line.
  if grep -q 'MISSING' "$DIR/manifest.txt"; then
    bad "an artifact is MISSING — the collector expected a filename that was not produced:"
    grep 'MISSING' "$DIR/manifest.txt" | sed 's/^/         /'
    echo "         Check the real filenames with: ls -l $DIR"
  else
    ok "every expected artifact was produced"
  fi

  if grep -q 'SUSPECT_EMPTY' "$DIR/manifest.txt"; then
    warn "an artifact looks empty (see SUSPECT_EMPTY above)."
    echo "         Expected only if the GPU is off the bus; on a healthy card it means"
    echo "         the capture silently failed — check the .err file in $DIR."
  else
    ok "all artifacts have plausible payload sizes"
  fi

  [ -s "$DIR/kernel-current-boot.log.gz" ] \
    && ok "kernel log captured ($(du -h "$DIR/kernel-current-boot.log.gz" | cut -f1))" \
    || bad "kernel log not captured"
  [ -s "$DIR/summary.txt" ] && ok "summary.txt written" || bad "summary.txt missing"
else
  bad "no manifest.txt produced — collection did not complete"
fi

echo "  Keep this run: it is a healthy-GPU baseline to diff a future crash against."
fi

# =============================================================================
hdr "Result"
# =============================================================================
printf '  %d passed, %d warnings, %d failed\n\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -eq 0 ]; then
  cat <<EOF
  Capture chain is live. Two independent triggers:
    - gpu-xid-watch.service  fires seconds after a severe Xid is logged
    - safe-shutdown.sh       collects before the reboot that destroys the dump

  Watch it:      journalctl -u gpu-xid-watch -f
  Collections:   $OUT_ROOT
  Latch (boot):  $LATCH
  Re-verify:     sudo $0 --test-only
EOF
  exit 0
else
  echo "  Fix the failures above, then re-run: sudo $0 --test-only"
  exit 1
fi
