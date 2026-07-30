#!/bin/bash
# GuideLLM FIXED-CONCURRENCY benchmark for Laguna-S-2.1-NVFP4 (117.6B MoE, 8.5B active).
#
# Same regime as models/qwen3.6/bench-guidellm-concurrent.sh: a pinned
# concurrency ladder rather than --profile sweep. The sweep profile's throughput
# strategy dispatches 512 concurrent requests and, on long scenarios, nothing
# completes inside the per-rate window (0 successful requests). A fixed ladder
# gives a clean, server-capacity-bounded point at each concurrency.
#
# KV BUDGET — the reason the server config matters here:
#   At --gpu-memory-utilization 0.90 the KV pool is 448,413 tokens. The agentic
#   scenario is ~16.8k tokens/request, so concurrency 32 needs ~538k tokens and
#   would measure KV starvation instead of concurrency scaling. Serve this with
#   GPU_MEM_UTIL=0.94 and MAX_NUM_SEQS=64 (the compose defaults) so the ladder
#   is compute-bound rather than KV-bound. Scenarios still expected to exceed
#   the pool at rate 64 are flagged in the table below — that is real saturation
#   behaviour, but report it as KV-bound, not as a scaling result.
#
# Approx tokens in flight = rate x (prompt + output):
#   scenario       per-req    @16      @32      @64
#   chat            2.3k      37k      74k     147k
#   codegen         5.5k      88k     176k     352k
#   rag             8.3k     133k     266k     531k
#   summarization  12.3k     197k     394k     787k  <- exceeds pool at 64
#   agentic        16.8k     269k     538k    1075k  <- exceeds pool at 32 @0.90
#
# Usage:
#   ./bench-guidellm-concurrent.sh                 # all scenarios
#   ./bench-guidellm-concurrent.sh chat codegen    # subset
# Override via env: TARGET, MODEL, PROCESSOR, OUTPUT_DIR, RATES, MAX_SECONDS.

set -euo pipefail

TARGET="${TARGET:-http://localhost:11440}"
MODEL="${MODEL:-laguna-s-2.1}"
PROCESSOR="${PROCESSOR:-poolside/Laguna-S-2.1-NVFP4}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/benchmarks/concurrent}"
RATES="${RATES:-1,4,8,16,32,64}"
MAX_SECONDS="${MAX_SECONDS:-90}"
WARMUP="${WARMUP:-0.15}"

# name|prompt_tokens|prompt_stdev|output_tokens|output_stdev|description
SCENARIOS=(
  "chat|2000|500|300|75|Multi-turn conversation (balanced, latency-sensitive)"
  "rag|8000|1500|256|64|RAG with 4-6 retrieved chunks (prefill-heavy)"
  "agentic|16000|3000|800|200|Tool-use agent: system prompt + tools + history"
  "codegen|4000|1000|1500|400|Code generation: file context to full functions"
  "summarization|12000|2000|300|75|Document summarization (pure prefill stress)"
)

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

curl -sf --max-time 5 "${TARGET}/health" >/dev/null || { echo "ERROR: ${TARGET} not healthy"; exit 1; }
mkdir -p "${OUTPUT_DIR}"

log "target=${TARGET} model=${MODEL} rates=${RATES} max_seconds=${MAX_SECONDS}"

selected=("$@")
for entry in "${SCENARIOS[@]}"; do
  IFS='|' read -r name pt ps ot os desc <<< "${entry}"
  if [[ ${#selected[@]} -gt 0 ]]; then
    found=false; for s in "${selected[@]}"; do [[ "$s" == "$name" ]] && found=true; done
    [[ "$found" == false ]] && continue
  fi
  data_spec="prompt_tokens=${pt},prompt_tokens_stdev=${ps},output_tokens=${ot},output_tokens_stdev=${os}"
  scenario_dir="${OUTPUT_DIR}/${name}"; mkdir -p "${scenario_dir}"
  log "━━ ${name}: ${desc}"
  log "   concurrency ladder=${RATES}  max_seconds=${MAX_SECONDS}  (${pt}±${ps} in / ${ot}±${os} out)"
  guidellm benchmark \
    --target "${TARGET}" --model "${MODEL}" --processor "${PROCESSOR}" \
    --profile concurrent --rate "${RATES}" \
    --data "${data_spec}" \
    --max-seconds "${MAX_SECONDS}" --warmup "${WARMUP}" \
    --output-dir "${scenario_dir}" --outputs json,csv \
    2>&1 | tee "${scenario_dir}/console.log"
  log "   saved -> ${scenario_dir}/"
done
log "All concurrent benchmarks complete: ${OUTPUT_DIR}/"
