#!/bin/bash
# Three-way A/B/C benchmark on vLLM v0.23.0-cu129 (engine held constant):
#   A  unsloth/Qwen3.6-27B-NVFP4 + MTP self-speculation (the newly re-uploaded ckpt)
#   B  Qwen3.6-27B-FP8 + DFlash (z-lab drafter, 15 tok) — production baseline
#   C  sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP + MTP (existing NVFP4+MTP rival)
# Each: bring up, wait healthy (crash-loop guard), test_chat (3 runs, warmup,
# no-think); tool-calling smoke on the candidate. One model at a time on the GPU.
set -u
cd /home/despara/Development/ai_services/models/qwen3.6 || exit 1
RES=benchmarks/v0230_3way_results.txt
PORT=11436
: > "$RES"
echo "vLLM v0.23.0-cu129 — Qwen3.6-27B 3-way — $(date -u)" | tee -a "$RES"

run() {
  name=$1; file=$2; cont=$3; tools=$4
  echo "" | tee -a "$RES"
  echo "==================== $name ====================" | tee -a "$RES"
  echo "compose: $file" | tee -a "$RES"
  docker compose --env-file ../../.env -f "$file" up -d >/dev/null 2>&1
  ok=0
  for i in $(seq 1 200); do   # up to ~33 min (covers first-time weight download)
    st=$(docker inspect --format='{{.State.Status}}' "$cont" 2>/dev/null)
    rc=$(docker inspect --format='{{.RestartCount}}' "$cont" 2>/dev/null)
    h=$(curl -fs -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)
    [ -z "$st" ] && { echo "[$name] container GONE" | tee -a "$RES"; break; }
    [ "$h" = 200 ] && { ok=1; echo "[$name] HEALTHY (~$((i*10))s)" | tee -a "$RES"; break; }
    if [ "$st" = restarting ] || [ "$st" = exited ] || [ "${rc:-0}" -gt 0 ]; then
      echo "[$name] CRASH-LOOP status=$st RestartCount=$rc" | tee -a "$RES"
      docker logs "$cont" 2>&1 | grep -iE 'error|traceback|valueerror|runtimeerror|assert|out of memory|no kernel|not supported|speculativ' | tail -15 | tee -a "$RES"
      break
    fi
    sleep 10
  done
  if [ "$ok" = 1 ]; then
    echo "--- test_chat (3 runs, warmup, no-think) ---" | tee -a "$RES"
    python3 ../shared/test_chat.py --base-url http://localhost:$PORT/v1 --model qwen3.6-27b --runs 3 --warmup --no-think 2>&1 | tee -a "$RES"
    if [ "$tools" = "yes" ]; then
      echo "--- test_tools (single scenario) ---" | tee -a "$RES"
      python3 ../shared/test_tools.py --base-url http://localhost:$PORT/v1 --model qwen3.6-27b --scenario single 2>&1 | tail -25 | tee -a "$RES"
    fi
  fi
  docker compose -f "$file" down >/dev/null 2>&1
  sleep 5
}

# free the GPU: remove the live container so compose B can recreate it cleanly
docker stop qwen36-27b-fp8-dflash >/dev/null 2>&1
docker rm   qwen36-27b-fp8-dflash >/dev/null 2>&1

run "A unsloth NVFP4+MTP"   docker-compose.vllm-27b-nvfp4-unsloth-rtx.yml qwen36-27b-nvfp4-unsloth yes
run "B FP8+DFlash"          docker-compose.vllm-27b-fp8-dflash-rtx.yml    qwen36-27b-fp8-dflash    no
run "C sakamaki NVFP4+MTP"  docker-compose.vllm-27b-nvfp4-mtp-rtx.yml     qwen36-27b-nvfp4-mtp     no

echo "" | tee -a "$RES"; echo "ALL DONE $(date -u)" | tee -a "$RES"
