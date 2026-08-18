#!/usr/bin/env bash
# Single-variable sweep harness for qwen3.8 vLLM configs.
#
# Usage:   ./sweep.sh <label> [extra vllm args...]
# Env:     SWEEP_MODEL=<hf id>      target checkpoint (default: gittensor W4A4)
#          SWEEP_ENV="-e A=1 -e B=2"  extra docker -e flags
#
# Records BOTH numbers, because they answer different questions:
#   sampled : ../shared/test_chat.py, model's own generation_config
#             (temperature=1.0, thinking on) — how fast the endpoint really is.
#   greedy  : ./bench_greedy.py, temperature=0 + thinking off + ignore_eos —
#             the protocol vendor checkpoint cards use, so our numbers are
#             comparable to theirs. Speculative acceptance is much higher at
#             temperature 0, so greedy > sampled for every spec config.
#
# EVERY benchmark call is wrapped in `timeout`. This is not paranoia: the
# combination --async-scheduling + draft_sample_method=probabilistic HANGS this
# vLLM build — the API server answers, the engine allocates 90 GB, GPU
# utilisation sits at 0%, and /health never returns. Without a timeout the
# harness waits forever and takes the whole sweep with it.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ../../.env 2>/dev/null; set +a
LABEL="$1"; shift
MODEL="${SWEEP_MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
OUT="benchmarks/sweep-${LABEL}.txt"
mkdir -p benchmarks

record() { echo "RESULT|$LABEL|$1"; echo "RESULT|$LABEL|$1" >> benchmarks/RESULTS.log; }
# SIGTERM with a long grace period, never `rm -f`: SIGKILL mid-CUDA-op is what
# wedges this card badly enough to need a cold power cycle (see CLAUDE.md).
cleanup() { docker stop -t 90 sweep >/dev/null 2>&1; docker rm sweep >/dev/null 2>&1; }
cleanup

docker run -d --name sweep --gpus all --ipc=host --shm-size 64gb \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 11484:8000 \
  -v qwen38_huggingface_cache:/root/.cache/huggingface \
  -v "$PWD/drafters":/drafters:ro \
  -e HF_TOKEN="$HF_TOKEN" -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
  ${SWEEP_ENV:-} \
  vllm/vllm-openai:qwen38-x86_64-cu130 \
  "$MODEL" --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 262144 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  "$@" >/dev/null 2>&1 || { record "RUN-FAILED"; exit 1; }

t0=$(date +%s)
while true; do
  st=$(docker inspect --format='{{.State.Status}}' sweep 2>/dev/null)
  [ -z "$st" ] && { record "GONE"; exit 1; }
  if [ "$st" != running ]; then
    record "DIED($st)"
    docker logs sweep 2>&1 | grep -iE 'error|valueerror|runtimeerror|assert|out of memory|not supported|unsupported' | tail -8
    docker logs sweep > "$OUT" 2>&1; cleanup; exit 1
  fi
  curl -sf --max-time 10 http://localhost:11484/health >/dev/null 2>&1 && break
  if [ $(( $(date +%s)-t0 )) -gt 1200 ]; then record "BOOT-TIMEOUT"; docker logs sweep > "$OUT" 2>&1; cleanup; exit 1; fi
  sleep 8
done
BOOT=$(( $(date +%s)-t0 ))

{ echo "### $LABEL"; echo "model: $MODEL"; echo "env: ${SWEEP_ENV:-} FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}";
  echo "args: $*"; echo "boot: ${BOOT}s"; date -Is; echo; } > "$OUT"

echo "--- sampled (test_chat.py, temp=1.0, thinking on) ---" >> "$OUT"
timeout 1200 python3 ../shared/test_chat.py --base-url http://localhost:11484/v1 \
    --model qwen3.8-27b --runs 7 --warmup >> "$OUT" 2>&1
[ $? -eq 124 ] && echo "HUNG (timeout 600s)" >> "$OUT"

echo "--- greedy (bench_greedy.py, temp=0, thinking off, ignore_eos 256) ---" >> "$OUT"
timeout 420 python3 ./bench_greedy.py --base-url http://localhost:11484/v1 \
    --model qwen3.8-27b --runs 5 --output-tokens 256 >> "$OUT" 2>&1
[ $? -eq 124 ] && echo "GREEDY HUNG (timeout 420s)" >> "$OUT"

{ echo "--- spec metrics ---"
  docker logs sweep 2>&1 | grep -oE 'Mean acceptance length: [0-9.]+.*' | tail -4
  echo "--- engine ---"
  docker logs sweep 2>&1 | grep -iE 'GPU KV cache size|NVFP4 GEMM|GDN decode kernel|model weights take' | tail -5
} >> "$OUT"

SAMP=$(grep "Average  tok/s" "$OUT" | awk '{print $3}')
GRD=$(grep "GREEDY avg tok/s" "$OUT" | awk '{print $4}')
TT=$(grep "Average  TTFT" "$OUT" | awk '{print $4}')
ACC=$(docker logs sweep 2>&1 | grep -oE 'Mean acceptance length: [0-9.]+' | tail -8 | awk '{s+=$4;n++} END{if(n)printf "%.2f",s/n}')
grep -q HUNG "$OUT" && HUNGF="HUNG " || HUNGF=""
record "${HUNGF}sampled=${SAMP:-FAIL}|greedy=${GRD:-FAIL}|ttft=${TT:-?}ms|accept=${ACC:-n/a}|boot=${BOOT}s"
cleanup
