#!/bin/bash
# GuideLLM FIXED-CONCURRENCY benchmark for Qwen3.6-27B (parallelization study).
#
# Complements bench-guidellm-parallel.sh (which uses --profile sweep). The sweep
# profile's throughput strategy dispatches 512 concurrent requests and, for long
# scenarios, nothing completes inside the per-rate window (0 successful requests).
# This script pins concurrency to a fixed ladder so each point is a clean,
# server-capacity-bounded measurement — the right shape for a scaling curve.
#
# Usage:
#   ./bench-guidellm-concurrent.sh                 # all scenarios
#   ./bench-guidellm-concurrent.sh chat codegen    # subset
# Override via env: TARGET, MODEL, PROCESSOR, OUTPUT_DIR, RATES, MAX_SECONDS.

set -euo pipefail

TARGET="${TARGET:-http://localhost:11436}"
MODEL="${MODEL:-qwen3.6-27b}"
PROCESSOR="${PROCESSOR:-Qwen/Qwen3.6-27B-FP8}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/benchmarks/concurrent}"
RATES="${RATES:-1,4,8,16,32}"
MAX_SECONDS="${MAX_SECONDS:-75}"
WARMUP="${WARMUP:-0.15}"

# name|prompt_tokens|prompt_stdev|output_tokens|output_stdev|description
SCENARIOS=(
  "chat|2000|500|300|75|Multi-turn conversation (balanced, latency-sensitive)"
  "rag|8000|1500|256|64|RAG with 4-6 retrieved chunks (prefill-heavy)"
  "agentic|16000|3000|800|200|Tool-use agent: system prompt + tools + history"
  "codegen|4000|1000|1500|400|Code generation: file context to full functions"
)

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

curl -sf --max-time 5 "${TARGET}/health" >/dev/null || { echo "ERROR: ${TARGET} not healthy"; exit 1; }
mkdir -p "${OUTPUT_DIR}"

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
