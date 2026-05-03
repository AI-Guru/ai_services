#!/usr/bin/env bash
# Sweep DFlash num_speculative_tokens on the INT4 target.
# Restarts the int4-dflash container for each N, then runs test_chat.py
# (3 runs, --warmup, --no-think). Records avg + steady-state to a TSV.
#
# Usage: bash sweep_dflash_n.sh "6 8 10 12 18 20"
set -euo pipefail
cd "$(dirname "$0")"

VALUES=${1:-"6 8 10 12 18 20"}
COMPOSE=docker-compose.vllm-27b-int4-dflash-rtx.yml
RESULTS=/tmp/dflash_n_sweep.tsv
echo -e "N\tavg_tok_s\trun1\trun2\trun3\tavg_ttft_s" > "$RESULTS"

for N in $VALUES; do
  echo ""
  echo "==================== N = $N ===================="
  # Patch the speculative-config line in place.
  sed -i -E "s|(\"num_speculative_tokens\":)[0-9]+|\1${N}|" "$COMPOSE"

  docker compose --env-file ../../.env -f "$COMPOSE" down 2>&1 | tail -3
  docker compose --env-file ../../.env -f "$COMPOSE" up -d 2>&1 | tail -3

  echo "Waiting for /health…"
  until curl -fs http://localhost:11436/health > /dev/null 2>&1; do sleep 5; done
  echo "READY (N=$N)"

  RAW=$(python3 ../shared/test_chat.py --base-url http://localhost:11436/v1 \
           --model qwen3.6-27b --runs 3 --warmup --no-think 2>&1)

  # Strip ANSI for parsing.
  CLEAN=$(printf '%s\n' "$RAW" | sed -E 's/\x1b\[[0-9;]*[mGKH]//g')

  AVG_TPS=$(printf '%s\n' "$CLEAN" | awk '/Average +tok\/s:/ {print $NF}')
  AVG_TTFT=$(printf '%s\n' "$CLEAN" | awk '/Average +TTFT/ {print $4}')
  RUNS=$(printf '%s\n' "$CLEAN" | grep -E "TTFT[[:space:]]+[0-9].*tok/s" | head -3 \
            | awk -F'|' '{gsub(/[^0-9.]/,"",$3); print $3}')
  R1=$(echo "$RUNS" | sed -n 1p)
  R2=$(echo "$RUNS" | sed -n 2p)
  R3=$(echo "$RUNS" | sed -n 3p)

  echo "N=$N  avg=$AVG_TPS  runs=[$R1 $R2 $R3]  ttft=$AVG_TTFT ms"
  echo -e "${N}\t${AVG_TPS}\t${R1}\t${R2}\t${R3}\t${AVG_TTFT}" >> "$RESULTS"
done

echo ""
echo "=================== SUMMARY ==================="
cat "$RESULTS"
