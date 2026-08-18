#!/usr/bin/env bash
# Decisive head-to-head. Same boot machinery as sweep.sh, three metrics:
#   sampled  test_chat.py            temp=1.0, thinking on, natural length
#   fixedlen bench_greedy.py -t 1.0  temp=1.0, thinking on, ignore_eos 512
#   greedy   bench_greedy.py         temp=0,  thinking off, ignore_eos 256
# fixedlen is the tie-breaker: it keeps the endpoint's REAL sampling
# distribution (which is what decides speculative acceptance) but removes the
# sampled test's dominant variance source - thinking length varying run to run.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ../../.env 2>/dev/null; set +a
LABEL="$1"; shift
MODEL="${SWEEP_MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
OUT="benchmarks/final-${LABEL}.txt"; mkdir -p benchmarks
cleanup() { docker stop -t 90 sweep >/dev/null 2>&1; docker rm sweep >/dev/null 2>&1; }
cleanup
docker run -d --name sweep --gpus all --ipc=host --shm-size 64gb \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 11484:8000 \
  -v qwen38_huggingface_cache:/root/.cache/huggingface -v "$PWD/drafters":/drafters:ro \
  -e HF_TOKEN="$HF_TOKEN" -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 ${SWEEP_ENV:-} \
  vllm/vllm-openai:qwen38-x86_64-cu130 \
  "$MODEL" --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder "$@" >/dev/null 2>&1 \
  || { echo "FINAL|$LABEL|RUN-FAILED" | tee -a benchmarks/RESULTS.log; exit 1; }
t0=$(date +%s)
while true; do
  st=$(docker inspect --format='{{.State.Status}}' sweep 2>/dev/null)
  [ -z "$st" ] && { echo "FINAL|$LABEL|GONE" | tee -a benchmarks/RESULTS.log; exit 1; }
  [ "$st" != running ] && { echo "FINAL|$LABEL|DIED" | tee -a benchmarks/RESULTS.log; docker logs sweep > "$OUT" 2>&1; cleanup; exit 1; }
  curl -sf --max-time 10 http://localhost:11484/health >/dev/null 2>&1 && break
  [ $(( $(date +%s)-t0 )) -gt 1200 ] && { echo "FINAL|$LABEL|BOOT-TIMEOUT" | tee -a benchmarks/RESULTS.log; cleanup; exit 1; }
  sleep 8
done
{ echo "### FINAL $LABEL"; echo "args: $*"; echo "env: ${SWEEP_ENV:-}"; date -Is; } > "$OUT"
timeout 1500 python3 ../shared/test_chat.py --base-url http://localhost:11484/v1 --model qwen3.8-27b --runs 7 --warmup >> "$OUT" 2>&1
timeout 900 python3 ./bench_greedy.py --base-url http://localhost:11484/v1 --model qwen3.8-27b --runs 5 --output-tokens 512 --temperature 1.0 --think >> "$OUT" 2>&1
timeout 600 python3 ./bench_greedy.py --base-url http://localhost:11484/v1 --model qwen3.8-27b --runs 5 --output-tokens 256 >> "$OUT" 2>&1
docker logs sweep 2>&1 | grep -oE 'Mean acceptance length: [0-9.]+.*' | tail -6 >> "$OUT"
S=$(grep "Average  tok/s" "$OUT" | awk '{print $3}')
F=$(grep "FIXEDLEN avg tok/s" "$OUT" | awk '{print $4}')
G=$(grep "GREEDY avg tok/s"  "$OUT" | awk '{print $4}')
echo "FINAL|$LABEL|sampled=${S:-FAIL}|fixedlen=${F:-FAIL}|greedy=${G:-FAIL}" | tee -a benchmarks/RESULTS.log
cleanup
