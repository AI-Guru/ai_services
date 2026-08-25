#!/usr/bin/env bash
# Long-context half of the scaling grid, split out because the first attempt
# died on a guidellm data-spec validation error (stdev=0) after the container
# had already been torn down. Same boot flags as scaling.sh.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ../../.env 2>/dev/null; set +a; export HF_TOKEN
LABEL="$1"; MODEL="$2"
cleanup(){ docker stop -t 90 prod >/dev/null 2>&1; docker rm prod >/dev/null 2>&1; }
cleanup
docker run -d --name prod --gpus all --ipc=host --shm-size 64gb \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 11484:8000 \
  -v qwen38_huggingface_cache:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm/vllm-openai:qwen38-x86_64-cu130 \
  "$MODEL" --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 262144 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.88 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic"}' \
  >/dev/null 2>&1 || { echo "SCALE|$LABEL-lc|RUN-FAILED" >> benchmarks/SCALING.log; exit 1; }
t0=$(date +%s)
while true; do
  st=$(docker inspect --format='{{.State.Status}}' prod 2>/dev/null)
  [ -z "$st" ] && { echo "SCALE|$LABEL-lc|GONE" >> benchmarks/SCALING.log; exit 1; }
  [ "$st" != running ] && { echo "SCALE|$LABEL-lc|DIED" >> benchmarks/SCALING.log
      docker logs prod 2>&1 | grep -iE 'error|out of memory|unsupported|assert' | tail -8; cleanup; exit 1; }
  curl -sf --max-time 10 http://localhost:11484/health >/dev/null 2>&1 && break
  [ $(( $(date +%s)-t0 )) -gt 1800 ] && { echo "SCALE|$LABEL-lc|BOOT-TIMEOUT" >> benchmarks/SCALING.log; cleanup; exit 1; }
  sleep 8
done
KV=$(docker logs prod 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+' | tail -1)
echo "== $LABEL-lc booted in $(( $(date +%s)-t0 ))s | $KV =="
# 16 x 100K and 10 x 150K both sit right at the KV pool ceiling - that is the
# point of these two levels, so keep them even if the engine has to queue.
MAXSEC=240 ./gllm.sh "$LABEL" lc100k "1,4,8,16"
MAXSEC=240 ./gllm.sh "$LABEL" lc150k "1,4,8,10"
docker logs prod > "benchmarks/serverlog-scaling-${LABEL}-lc.txt" 2>&1
ALIVE=$(docker inspect --format='{{.State.Status}}' prod 2>/dev/null)
ERR=$(grep -icE "out of memory|RuntimeError|AssertionError|EngineDeadError|Engine core" "benchmarks/serverlog-scaling-${LABEL}-lc.txt" 2>/dev/null || echo 0)
echo "SCALE|$LABEL-lc|$KV|end_status=$ALIVE|err_lines=$ERR|done" >> benchmarks/SCALING.log
cleanup
