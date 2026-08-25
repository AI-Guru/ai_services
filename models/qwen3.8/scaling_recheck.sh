#!/usr/bin/env bash
# Re-measure the two cells where a 60s window is too short to reach steady
# state: with 64 x 10K prompts in flight vLLM logs "Running: 6, Waiting: 58",
# so the window closes while the engine is still draining the prefill backlog
# and the successful-only metrics under-report. 240s instead.
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
  >/dev/null 2>&1 || { echo "SCALE|$LABEL-recheck|RUN-FAILED" >> benchmarks/SCALING.log; exit 1; }
t0=$(date +%s)
while true; do
  st=$(docker inspect --format='{{.State.Status}}' prod 2>/dev/null)
  [ -z "$st" ] && { echo "SCALE|$LABEL-recheck|GONE" >> benchmarks/SCALING.log; exit 1; }
  [ "$st" != running ] && { echo "SCALE|$LABEL-recheck|DIED" >> benchmarks/SCALING.log; cleanup; exit 1; }
  curl -sf --max-time 10 http://localhost:11484/health >/dev/null 2>&1 && break
  [ $(( $(date +%s)-t0 )) -gt 1800 ] && { echo "SCALE|$LABEL-recheck|BOOT-TIMEOUT" >> benchmarks/SCALING.log; cleanup; exit 1; }
  sleep 8
done
echo "== $LABEL-recheck booted in $(( $(date +%s)-t0 ))s =="
MAXSEC=240 ./gllm.sh "$LABEL-long240" pp10k "32,64"
MAXSEC=240 ./gllm.sh "$LABEL-long240" pp8k  "32,64"
echo "SCALE|$LABEL-recheck|end_status=$(docker inspect --format='{{.State.Status}}' prod 2>/dev/null)|done" >> benchmarks/SCALING.log
cleanup
