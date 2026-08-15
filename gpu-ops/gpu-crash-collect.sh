#!/usr/bin/env bash
#
# gpu-crash-collect.sh — persist NVIDIA GPU crash evidence before it is destroyed.
#
# Why this exists:
#   On a severe fault the driver prints:
#     "A GPU crash dump has been created. If possible, please run
#      nvidia-bug-report.sh as root to collect this data before the NVIDIA
#      kernel module is unloaded."
#   That dump lives in driver memory. It survives the fault itself and stays
#   readable for as long as the module stays loaded — but the reboot you perform
#   to recover the card destroys it. On 2026-08-14 (Xid 79, "GPU has fallen off
#   the bus", 18:36:05 -> reboot 18:38:55) it was lost exactly that way: the
#   evidence was sitting there for nearly three minutes and nothing collected it.
#
#   This script collects it. It is safe to call on every shutdown: when the
#   current boot logged no severe Xid it exits in well under a second.
#
# Usage:
#   sudo gpu-crash-collect.sh              # collect IFF this boot logged a severe Xid
#   sudo gpu-crash-collect.sh --full       # ... and also attempt the slow full bug report
#   sudo gpu-crash-collect.sh --force      # collect unconditionally (manual / testing)
#   sudo gpu-crash-collect.sh --check      # exit 0 if a severe Xid is present, else 1
#
# Called automatically from:
#   - gpu-xid-watch.service  — fires within seconds of the Xid being logged (--full)
#   - safe-shutdown.sh       — backstop, before the reboot that destroys the dump
#
set -uo pipefail

OUT_ROOT=/var/log/gpu-crash
LATCH=/run/gpu-crash-collected   # on tmpfs: cleared every boot => one collection per boot
KEEP=10                          # how many past collections to retain

# Timeouts. A wedged GPU makes every device query a potential hang, so nothing
# here is allowed to block indefinitely — this runs on the shutdown path.
# Measured 2026-08-14 on a HEALTHY card: safe report 25s, full report 31s,
# debugdump ~1s. The margins below are deliberately wider than that, because the
# only time this runs for real is when the GPU is misbehaving and every device
# query is a hang candidate — the healthy-path timing is the optimistic bound.
T_SAFE_REPORT=90                 # nvidia-bug-report.sh --safe-mode (measured 25s healthy)
T_FULL_REPORT=120                # nvidia-bug-report.sh (full; --full only)
T_DEBUGDUMP=30
T_QUERY=15                       # nvidia-smi / lspci snapshots

# Xid codes worth collecting for. Deliberately EXCLUDES 13 and 31: those are
# ordinary application-level graphics exceptions and MMU faults (this box logged
# ~10 of them from python3 across Aug 11-12 alone). Collecting on those would
# bury the real hardware events under megabytes of routine noise.
#   48  double-bit ECC error
#   63/64  ECC page retirement / row remap
#   74  NVLink / internal bus error
#   79  GPU has fallen off the bus      <- the 2026-08-14 fault
#   92  high single-bit ECC error rate
#   93/94/95  contained/uncontained ECC or unrecoverable error
#   154 GPU recovery action required (accompanies the above)
SEVERE_XIDS="48 63 64 74 79 92 93 94 95 154"

FULL=0; FORCE=0; CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --full)    FULL=1 ;;
    --force)   FORCE=1 ;;
    --check)   CHECK=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *)         echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

log() { printf '%s  gpu-crash-collect: %s\n' "$(date '+%H:%M:%S')" "$*"; }

# --- 1. Which severe Xids did this boot log? ----------------------------------
# Log line shape:  NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.
xid_codes_this_boot() {
  journalctl -k -b --no-pager 2>/dev/null \
    | sed -nE 's/.*NVRM: Xid \(PCI:[^)]*\): ([0-9]+).*/\1/p' \
    | sort -un
}

mapfile -t FOUND < <(
  xid_codes_this_boot | while read -r x; do
    case " $SEVERE_XIDS " in *" $x "*) echo "$x" ;; esac
  done
)

if [ "$CHECK" -eq 1 ]; then
  [ "${#FOUND[@]}" -gt 0 ] && exit 0 || exit 1
fi

if [ "$FORCE" -eq 0 ] && [ "${#FOUND[@]}" -eq 0 ]; then
  exit 0   # the normal path on every clean shutdown — say nothing, cost nothing
fi

if [ "$(id -u)" -ne 0 ]; then
  log "ERROR: need root to collect (nvidia-bug-report.sh requires it)."
  exit 1
fi

# --- 2. Latch ----------------------------------------------------------------
# The 2026-08-14 fault emitted ~250,000 NVRM lines in six seconds. Without an
# atomic latch the watcher would spawn a collector per matching line into a
# machine that is already in trouble. `set -o noclobber` makes the create-or-fail
# a single atomic operation, so exactly one caller wins the race.
if [ "$FORCE" -eq 0 ]; then
  if ! (set -o noclobber; echo "$$" >"$LATCH") 2>/dev/null; then
    log "Already collected this boot (pid $(cat "$LATCH" 2>/dev/null || echo '?')) — skipping."
    exit 0
  fi
fi

STAMP=$(date '+%Y%m%d-%H%M%S')
TAG=$( [ "${#FOUND[@]}" -gt 0 ] && printf 'xid%s' "$(IFS=-; echo "${FOUND[*]}")" || echo forced )
DIR="$OUT_ROOT/$STAMP-$TAG"
mkdir -p "$DIR" || { log "ERROR: cannot create $DIR"; exit 1; }

log "Severe Xid(s) this boot: ${FOUND[*]:-none (forced)}. Collecting into $DIR"

# run_step <name> <timeout> <command...> — never fails the script, always records
# what happened. A hang or a missing device must not abort the rest of the run.
run_step() {
  local name="$1" t="$2"; shift 2
  local rc
  timeout -k 5 "$t" "$@" >"$DIR/$name.out" 2>"$DIR/$name.err"
  rc=$?
  case $rc in
    0)   log "  $name: ok" ;;
    124) log "  $name: TIMED OUT after ${t}s (expected if the GPU is off the bus)" ;;
    *)   log "  $name: exit $rc" ;;
  esac
  printf '%s\trc=%s\ttimeout=%ss\n' "$name" "$rc" "$t" >>"$DIR/manifest.txt"
  # Drop empty stdout/stderr files so the directory stays readable.
  [ -s "$DIR/$name.out" ] || rm -f "$DIR/$name.out"
  [ -s "$DIR/$name.err" ] || rm -f "$DIR/$name.err"
  return 0
}

# A zero exit code is NOT evidence that anything was captured. Verified on this
# box: run without the required privileges, nvidia-debugdump prints
# "Insufficient Permissions" for every component, writes a 44-byte empty zip —
# and still exits 0. Left unchecked, that sails through the manifest looking like
# a successful collection. So check the payload, not just the return code.
check_payload() {
  local base="$1" min="$2" label="$3" f="" cand size
  # Resolve the real filename. nvidia-bug-report.sh takes --output-file <name>
  # and appends .gz itself when gzip is present, so <name> given here comes back
  # as <name>.gz — NOT <name>.log.gz (that shape is only its built-in default).
  # Verified 2026-08-14: checking the wrong name reported MISSING for an 11 MB
  # report that had been written correctly.
  for cand in "$base" "$base.gz" "$base.log.gz"; do
    [ -f "$cand" ] && { f="$cand"; break; }
  done
  if [ -z "$f" ]; then
    log "  WARNING: $label produced no file (tried: $base{,.gz,.log.gz})"
    printf '%s\tMISSING\n' "$label" >>"$DIR/manifest.txt"
    return 0
  fi
  label="$(basename "$f")"
  size=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$size" -lt "$min" ]; then
    log "  WARNING: $label is only ${size}B (< ${min}B) — capture almost certainly failed."
    log "           Expected if the GPU is off the bus; check the .err file next to it."
    printf '%s\tbytes=%s\tSUSPECT_EMPTY\n' "$label" "$size" >>"$DIR/manifest.txt"
  else
    printf '%s\tbytes=%s\tok\n' "$label" "$size" >>"$DIR/manifest.txt"
  fi
  return 0
}

# --- 3. Kernel-side evidence (always works; the GPU is not involved) ----------
{
  echo "collected:    $(date -Is)"
  echo "hostname:     $(hostname)"
  echo "severe xids:  ${FOUND[*]:-none (forced)}"
  echo "boot started: $(uptime -s 2>/dev/null)"
  echo "driver:       $(cat /proc/driver/nvidia/version 2>/dev/null | head -1)"
  echo
  echo "--- all Xid lines this boot ---"
  journalctl -k -b --no-pager 2>/dev/null | grep -E 'NVRM: Xid' || echo "(none)"
} >"$DIR/summary.txt"

# The full kernel log, compressed: the 2026-08-14 event produced ~250k lines and
# the surrounding NVRM cascade is the bulk of the diagnostic value.
journalctl -k -b --no-pager 2>/dev/null | gzip -c >"$DIR/kernel-current-boot.log.gz"

# --- 4. GPU-side evidence (may hang or fail if the card is gone) --------------
# --safe-mode first, always: it exists precisely to "disable certain queries that
# might hang the system", so it is the one most likely to return something when
# the card is wedged. The richer full report is attempted afterwards, and only
# when there is time budget for it (--full), never on the shutdown path.
run_step nvidia-bug-report-safe "$T_SAFE_REPORT" \
  nvidia-bug-report.sh --safe-mode --output-file "$DIR/nvidia-bug-report-safe"
check_payload "$DIR/nvidia-bug-report-safe" 4096 nvidia-bug-report-safe

run_step nvidia-debugdump "$T_DEBUGDUMP" \
  nvidia-debugdump --dumpall --file "$DIR/nvidia-debugdump.zip"

check_payload "$DIR/nvidia-debugdump.zip" 1024 nvidia-debugdump.zip

run_step nvidia-smi-q "$T_QUERY" nvidia-smi -q
run_step lspci        "$T_QUERY" lspci -vvv -s 01:00.0
run_step docker-ps    "$T_QUERY" docker ps -a

if [ "$FULL" -eq 1 ]; then
  run_step nvidia-bug-report-full "$T_FULL_REPORT" \
    nvidia-bug-report.sh --extra-system-data --output-file "$DIR/nvidia-bug-report-full"
  check_payload "$DIR/nvidia-bug-report-full" 4096 nvidia-bug-report-full
fi

# --- 5. Rotate ---------------------------------------------------------------
mapfile -t OLD < <(ls -1dt "$OUT_ROOT"/*/ 2>/dev/null | tail -n +$((KEEP + 1)))
if [ "${#OLD[@]}" -gt 0 ]; then
  log "Rotating out ${#OLD[@]} old collection(s) (keeping $KEEP)."
  rm -rf "${OLD[@]}"
fi

log "Done. Size: $(du -sh "$DIR" 2>/dev/null | cut -f1). Attach nvidia-bug-report-safe.gz to any NVIDIA RMA/support case."
exit 0
