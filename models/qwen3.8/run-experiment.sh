#!/usr/bin/env bash
# Boot a compose, wait crash-loop-aware, benchmark, record. Usage: run-experiment.sh <compose.yml> <label>
set -uo pipefail
cd "$(dirname "$0")"
F="$1"; LABEL="${2:-$1}"
OUT="benchmarks/${LABEL}.txt"
mkdir -p benchmarks
C=$(grep -m1 'container_name:' "$F" | awk '{print $2}')

echo "=== $LABEL  ($F -> $C) ==="
docker compose --env-file ../../.env -f "$F" up -d >/dev/null 2>&1 || { echo "COMPOSE-UP-FAILED"; exit 1; }
base=$(docker inspect --format='{{.RestartCount}}' "$C" 2>/dev/null || echo 0)
t0=$(date +%s)
while true; do
  st=$(docker inspect --format='{{.State.Status}}'        "$C" 2>/dev/null)
  h=$( docker inspect --format='{{.State.Health.Status}}' "$C" 2>/dev/null)
  rc=$(docker inspect --format='{{.RestartCount}}'        "$C" 2>/dev/null)
  [ -z "$st" ] && { echo "GONE"; exit 1; }
  [ "$h" = healthy ] && { echo "HEALTHY after $(( $(date +%s)-t0 ))s"; break; }
  if [ "$st" = restarting ] || [ "$st" = exited ] || [ "${rc:-0}" -gt "${base:-0}" ]; then
    echo "CRASH-LOOP: status=$st RestartCount=$rc"
    docker logs "$C" 2>&1 | grep -iE 'error|valueerror|runtimeerror|assert|no module or parameter|out of memory|not supported|unsupported' | tail -20
    docker compose -f "$F" down >/dev/null 2>&1
    exit 1
  fi
  [ $(( $(date +%s)-t0 )) -gt 1800 ] && { echo "TIMEOUT"; docker compose -f "$F" down >/dev/null 2>&1; exit 1; }
  sleep 10
done

{
echo "### $LABEL"
echo "compose: $F"
date -Is
echo "--- engine facts ---"
docker logs "$C" 2>&1 | grep -iE 'Using .*kernel|GDN decode|quantization|kv.cache.dtype|GPU KV cache size|Maximum concurrency|model weights take|Speculative|acceptance|graph capturing|torch.compile' | sed 's/^/  /' | tail -30
echo "--- test_chat.py --runs 3 --warmup ---"
} > "$OUT"

python3 ../shared/test_chat.py --base-url http://localhost:11484/v1 --model qwen3.8-27b --runs 3 --warmup 2>&1 | tee -a "$OUT" | tail -6
{
echo ""
echo "--- speculative metrics (post-run) ---"
docker logs "$C" 2>&1 | grep -iE 'Speculative metrics|acceptance|accepted|drafted' | tail -8 | sed 's/^/  /'
echo "--- prometheus spec counters ---"
curl -s http://localhost:11484/metrics 2>/dev/null | grep -E '^vllm:spec_decode' | grep -v '_bucket' | sed 's/^/  /'
echo ""
} >> "$OUT"
grep -E "Average" "$OUT"
echo "=== wrote $OUT ==="
