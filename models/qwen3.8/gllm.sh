#!/usr/bin/env bash
# GuideLLM concurrency sweep for one config+scenario.
# Usage: gllm.sh <label> <scenario> <rates>
# Env:   MAXSEC (default 60) - seconds per concurrency level.
# NEVER pipe guidellm through head/tail/tee - it SIGPIPE-deadlocks. Redirect only.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ../../.env 2>/dev/null; set +a; export HF_TOKEN
LABEL="$1"; SCEN="$2"; RATES="$3"
MAXSEC="${MAXSEC:-60}"
# guidellm aborts a request after backend timeout seconds and books it as an
# ERROR. Its default is 60s, which is fine up to ~10K prompts but silently
# shreds every long-context level (100K at conc 16: 33 errored / 4 ok).
GLLM_TIMEOUT="${GLLM_TIMEOUT:-60}"
# The pp* / lc* family is the 2026-08-22 scaling grid: prompt length is the only
# variable, output is pinned to 300 (132-500) so the rows are comparable.
TG="output_tokens=300,output_tokens_stdev=75,output_tokens_min=132,output_tokens_max=500"
case "$SCEN" in
  chat)    D="prompt_tokens=2000,prompt_tokens_stdev=500,prompt_tokens_min=1000,prompt_tokens_max=3000,output_tokens=300,output_tokens_stdev=75,output_tokens_min=150,output_tokens_max=450";;
  rag)     D="prompt_tokens=8000,prompt_tokens_stdev=1500,prompt_tokens_min=4000,prompt_tokens_max=12000,output_tokens=256,output_tokens_stdev=64,output_tokens_min=128,output_tokens_max=384";;
  agentic) D="prompt_tokens=16000,prompt_tokens_stdev=3000,prompt_tokens_min=8000,prompt_tokens_max=24000,output_tokens=800,output_tokens_stdev=200,output_tokens_min=400,output_tokens_max=1200";;
  codegen) D="prompt_tokens=4000,prompt_tokens_stdev=1000,prompt_tokens_min=2000,prompt_tokens_max=6000,output_tokens=1500,output_tokens_stdev=400,output_tokens_min=750,output_tokens_max=2250";;
  summ)    D="prompt_tokens=12000,prompt_tokens_stdev=2000,prompt_tokens_min=6000,prompt_tokens_max=18000,output_tokens=300,output_tokens_stdev=75,output_tokens_min=150,output_tokens_max=450";;
  pp512)   D="prompt_tokens=512,prompt_tokens_stdev=128,prompt_tokens_min=256,prompt_tokens_max=768,$TG";;
  pp2k)    D="prompt_tokens=2000,prompt_tokens_stdev=400,prompt_tokens_min=1200,prompt_tokens_max=2800,$TG";;
  pp4k)    D="prompt_tokens=4000,prompt_tokens_stdev=800,prompt_tokens_min=2400,prompt_tokens_max=5600,$TG";;
  pp8k)    D="prompt_tokens=8000,prompt_tokens_stdev=1600,prompt_tokens_min=4800,prompt_tokens_max=11200,$TG";;
  pp10k)   D="prompt_tokens=10000,prompt_tokens_stdev=2000,prompt_tokens_min=6000,prompt_tokens_max=14000,$TG";;
  lc100k)  D="prompt_tokens=100000,prompt_tokens_stdev=1,prompt_tokens_min=99000,prompt_tokens_max=101000,$TG";;
  lc150k)  D="prompt_tokens=150000,prompt_tokens_stdev=1,prompt_tokens_min=149000,prompt_tokens_max=151000,$TG";;
  *) echo "unknown scenario $SCEN"; exit 1;;
esac
O="benchmarks/guidellm/${LABEL}-${SCEN}"
mkdir -p "$O"
timeout 5400 guidellm benchmark run \
  --target http://localhost:11484 --model qwen3.8-27b \
  --processor Qwen/Qwen3.8-27B --request-type chat_completions \
  --profile concurrent --rate "$RATES" --data "$D" \
  --backend-kwargs "{\"timeout\": $GLLM_TIMEOUT}" \
  --max-seconds "$MAXSEC" --warmup 0.1 --disable-progress \
  --output-dir "$O" --outputs json > "$O/console.txt" 2>&1
echo "exit=$? -> $O"
python3 summarize_gllm.py "$O/benchmarks.json" "$LABEL" "$SCEN"
