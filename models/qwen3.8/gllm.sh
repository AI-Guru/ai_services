#!/usr/bin/env bash
# GuideLLM concurrency sweep for one config+scenario.
# Usage: gllm.sh <label> <scenario> <rates>
# NEVER pipe guidellm through head/tail/tee - it SIGPIPE-deadlocks. Redirect only.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ../../.env 2>/dev/null; set +a; export HF_TOKEN
LABEL="$1"; SCEN="$2"; RATES="$3"
case "$SCEN" in
  chat)    D="prompt_tokens=2000,prompt_tokens_stdev=500,prompt_tokens_min=1000,prompt_tokens_max=3000,output_tokens=300,output_tokens_stdev=75,output_tokens_min=150,output_tokens_max=450";;
  rag)     D="prompt_tokens=8000,prompt_tokens_stdev=1500,prompt_tokens_min=4000,prompt_tokens_max=12000,output_tokens=256,output_tokens_stdev=64,output_tokens_min=128,output_tokens_max=384";;
  agentic) D="prompt_tokens=16000,prompt_tokens_stdev=3000,prompt_tokens_min=8000,prompt_tokens_max=24000,output_tokens=800,output_tokens_stdev=200,output_tokens_min=400,output_tokens_max=1200";;
  codegen) D="prompt_tokens=4000,prompt_tokens_stdev=1000,prompt_tokens_min=2000,prompt_tokens_max=6000,output_tokens=1500,output_tokens_stdev=400,output_tokens_min=750,output_tokens_max=2250";;
  summ)    D="prompt_tokens=12000,prompt_tokens_stdev=2000,prompt_tokens_min=6000,prompt_tokens_max=18000,output_tokens=300,output_tokens_stdev=75,output_tokens_min=150,output_tokens_max=450";;
  *) echo "unknown scenario $SCEN"; exit 1;;
esac
O="benchmarks/guidellm/${LABEL}-${SCEN}"
mkdir -p "$O"
timeout 3000 guidellm benchmark run \
  --target http://localhost:11484 --model qwen3.8-27b \
  --processor Qwen/Qwen3.8-27B --request-type chat_completions \
  --profile concurrent --rate "$RATES" --data "$D" \
  --max-seconds 60 --warmup 0.1 --disable-progress \
  --output-dir "$O" --outputs json > "$O/console.txt" 2>&1
echo "exit=$? -> $O"
python3 summarize_gllm.py "$O/benchmarks.json" "$LABEL" "$SCEN"
