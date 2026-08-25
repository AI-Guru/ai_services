#!/usr/bin/env bash
# Boot the SGLang + DFlash2 compose and run the same three-metric protocol as
# final.sh, so the numbers land in benchmarks/RESULTS.log directly comparable to
# the vLLM rows:
#   sampled  ../shared/test_chat.py     temp=1.0, thinking on, natural length
#   fixedlen ./bench_greedy.py -t 1.0   temp=1.0, thinking on, ignore_eos 512
#   greedy   ./bench_greedy.py          temp=0,  thinking off, ignore_eos 256
#
# Boot waiting is crash-loop aware per CLAUDE.md even though restart is "no":
# a DFLASH/flashinfer mismatch exits rather than loops, and this reports it
# instead of sitting out the full timeout.
set -uo pipefail
cd "$(dirname "$0")"
# Usage: sglang_dflash_bench.sh [label] [compose-file] [container-name]
LABEL="${1:-sglang-dflash2}"
F="${2:-docker-compose.sglang-27b-nvfp4-dflash2-rtx.yml}"
C="${3:-qwen38-27b-sglang-dflash2}"
URL=http://localhost:11485
OUT="benchmarks/final-${LABEL}.txt"; mkdir -p benchmarks

# SIGTERM + long grace, never `rm -f` — SIGKILL mid-CUDA-op wedges this card.
cleanup() { docker compose --env-file ../../.env -f "$F" down -t 90 >/dev/null 2>&1; }
cleanup
docker compose --env-file ../../.env -f "$F" up -d >/dev/null 2>&1 \
  || { echo "FINAL|$LABEL|RUN-FAILED" | tee -a benchmarks/RESULTS.log; exit 1; }

t0=$(date +%s)
while true; do
  st=$(docker inspect --format='{{.State.Status}}' "$C" 2>/dev/null)
  [ -z "$st" ] && { echo "FINAL|$LABEL|GONE" | tee -a benchmarks/RESULTS.log; exit 1; }
  if [ "$st" = exited ] || [ "$st" = restarting ]; then
    echo "FINAL|$LABEL|DIED($st)" | tee -a benchmarks/RESULTS.log
    docker logs "$C" > "$OUT" 2>&1; cleanup; exit 1
  fi
  curl -sf --max-time 10 "$URL/health" >/dev/null 2>&1 && break
  [ $(( $(date +%s)-t0 )) -gt 2400 ] && { echo "FINAL|$LABEL|BOOT-TIMEOUT" | tee -a benchmarks/RESULTS.log; docker logs "$C" > "$OUT" 2>&1; cleanup; exit 1; }
  sleep 8
done
BOOT=$(( $(date +%s)-t0 ))

{ echo "### FINAL $LABEL"; echo "file: $F"; echo "boot: ${BOOT}s"; date -Is; } > "$OUT"
timeout 1500 python3 ../shared/test_chat.py --base-url "$URL/v1" --model qwen3.8-27b --runs 7 --warmup >> "$OUT" 2>&1
timeout 900  python3 ./bench_greedy.py --base-url "$URL/v1" --model qwen3.8-27b --runs 5 --output-tokens 512 --temperature 1.0 --think >> "$OUT" 2>&1
timeout 600  python3 ./bench_greedy.py --base-url "$URL/v1" --model qwen3.8-27b --runs 5 --output-tokens 256 >> "$OUT" 2>&1

docker logs "$C" 2>&1 | grep -oE '(accept|Accept)[a-z ]*(length|rate)[: ]+[0-9.]+' | tail -6 >> "$OUT"
docker logs "$C" > "benchmarks/serverlog-${LABEL}.txt" 2>&1

S=$(grep "Average  tok/s"    "$OUT" | awk '{print $3}')
F2=$(grep "FIXEDLEN avg tok/s" "$OUT" | awk '{print $4}')
G=$(grep "GREEDY avg tok/s"   "$OUT" | awk '{print $4}')
echo "FINAL|$LABEL|sampled=${S:-FAIL}|fixedlen=${F2:-FAIL}|greedy=${G:-FAIL}|boot=${BOOT}s" | tee -a benchmarks/RESULTS.log
