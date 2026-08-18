#!/usr/bin/env bash
# Production-readiness test for ONE config: boot -> chat concurrency sweep ->
# mixed text+image load -> teardown.
# Usage: prodtest.sh <label> [vllm args...]
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ../../.env 2>/dev/null; set +a; export HF_TOKEN
LABEL="$1"; shift
MODEL="${SWEEP_MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
cleanup(){ docker stop -t 90 prod >/dev/null 2>&1; docker rm prod >/dev/null 2>&1; }
cleanup
docker run -d --name prod --gpus all --ipc=host --shm-size 64gb \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 11484:8000 \
  -v qwen38_huggingface_cache:/root/.cache/huggingface -v "$PWD/drafters":/drafters:ro \
  -e HF_TOKEN="$HF_TOKEN" -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 ${SWEEP_ENV:-} \
  vllm/vllm-openai:qwen38-x86_64-cu130 \
  "$MODEL" --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder "$@" >/dev/null 2>&1 \
  || { echo "PROD|$LABEL|RUN-FAILED" >> benchmarks/PROD.log; exit 1; }
t0=$(date +%s)
while true; do
  st=$(docker inspect --format='{{.State.Status}}' prod 2>/dev/null)
  [ -z "$st" ] && { echo "PROD|$LABEL|GONE" >> benchmarks/PROD.log; exit 1; }
  [ "$st" != running ] && { echo "PROD|$LABEL|DIED" >> benchmarks/PROD.log
      docker logs prod 2>&1 | grep -iE 'error|out of memory|unsupported' | tail -5
      cleanup; exit 1; }
  curl -sf --max-time 10 http://localhost:11484/health >/dev/null 2>&1 && break
  [ $(( $(date +%s)-t0 )) -gt 1200 ] && { echo "PROD|$LABEL|BOOT-TIMEOUT" >> benchmarks/PROD.log; cleanup; exit 1; }
  sleep 8
done
KV=$(docker logs prod 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+' | tail -1)
echo "== $LABEL booted in $(( $(date +%s)-t0 ))s | $KV =="
./gllm.sh "$LABEL" chat "1,2,4,8,16,32,64"
for C in 8 32; do
  python3 mixed_load.py --concurrency $C --requests $((C*4)) --label "$LABEL" 2>&1 | tail -7
done
# Always keep the server log: a config that dies mid-sweep looks identical to a
# harness failure from the client side (guidellm reports "worker process group
# startup failed", mixed_load reports Connection refused). The engine log is the
# only thing that distinguishes them.
docker logs prod > "benchmarks/serverlog-${LABEL}.txt" 2>&1
ALIVE=$(docker inspect --format='{{.State.Status}}' prod 2>/dev/null)
CRASH=$(grep -icE "out of memory|RuntimeError|AssertionError|EngineDeadError|Engine core" "benchmarks/serverlog-${LABEL}.txt" 2>/dev/null || echo 0)
echo "PROD|$LABEL|$KV|end_status=$ALIVE|err_lines=$CRASH" >> benchmarks/PROD.log
cleanup
