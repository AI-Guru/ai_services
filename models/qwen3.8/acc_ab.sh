#!/usr/bin/env bash
# Boot one ad-hoc SGLang variant on 11488, run GSM8K, tear down.
# Usage: acc_ab.sh "<label>" [extra sglang flags...]
set -uo pipefail
cd "$(dirname "$0")"
PQ=/tmp/claude-1000/-home-despara-Development-ai-services/3ad23d93-782f-401a-90aa-5af3808c4e06/scratchpad/gsm8k_test.parquet
LABEL="$1"; shift
C=accprobe
docker rm -f $C >/dev/null 2>&1
docker run -d --name $C --gpus all --ipc=host --shm-size 32gb \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 11488:11488 \
  -v qwen38_huggingface_cache:/root/.cache/huggingface \
  --entrypoint sglang lmsysorg/sglang:dev-qwen38-27b-dflash2 serve \
  --trust-remote-code --model-path RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead \
  --served-model-name qwen3.8-27b --mem-fraction-static 0.85 \
  --attention-backend flashinfer --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 11.01 --mamba-radix-cache-strategy extra_buffer_lazy \
  --speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 8 --host 0.0.0.0 --port 11488 "$@" >/dev/null || {
    echo "ACC|$LABEL|RUN-FAILED" | tee -a benchmarks/ACCURACY.log; exit 1; }
t0=$(date +%s)
while true; do
  st=$(docker inspect --format='{{.State.Status}}' $C 2>/dev/null)
  [ -z "$st" ] && { echo "ACC|$LABEL|GONE" | tee -a benchmarks/ACCURACY.log; exit 1; }
  [ "$st" != running ] && { echo "ACC|$LABEL|DIED" | tee -a benchmarks/ACCURACY.log; docker logs $C 2>&1 | tail -20; docker rm -f $C >/dev/null; exit 1; }
  curl -sf --max-time 10 http://localhost:11488/health >/dev/null 2>&1 && break
  [ $(( $(date +%s)-t0 )) -gt 1200 ] && { echo "ACC|$LABEL|BOOT-TIMEOUT" | tee -a benchmarks/ACCURACY.log; docker rm -f $C >/dev/null; exit 1; }
  sleep 8
done
docker logs $C 2>&1 | tr '\r' '\n' | grep -E 'max_total_num_tokens=' | tail -1
timeout 2400 python3 gsm8k_eval.py --base-url http://localhost:11488/v1 --model qwen3.8-27b \
  --parquet "$PQ" --n ${ACC_N:-300} --concurrency 24 --label "$LABEL" | tee -a benchmarks/ACCURACY.log
docker stop -t 90 $C >/dev/null 2>&1; docker rm $C >/dev/null 2>&1
