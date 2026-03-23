#!/bin/bash
# GuideLLM parallel benchmarking for Nemotron models on vLLM
# Tests GPU parallelization across different workload profiles on RTX PRO 6000
#
# Token counts based on production data:
#   - OpenRouter State of AI 2025
#   - Databricks inference benchmarks
#   - Real agentic system traces
#
# Usage:
#   ./bench-guidellm-parallel.sh              # Run all scenarios
#   ./bench-guidellm-parallel.sh chat         # Run single scenario
#   ./bench-guidellm-parallel.sh chat rag     # Run specific scenarios
#
# Environment variables:
#   TARGET    — vLLM endpoint (default: http://localhost:11440)
#   MODEL     — served model name (default: nemotron-nano-30b)
#   PROCESSOR — HF tokenizer (default: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
#   SUFFIX    — output directory suffix (default: none, e.g. "-fp8", "-nvfp4")

set -euo pipefail

TARGET="${TARGET:-http://localhost:11440}"
MODEL="${MODEL:-nemotron-nano-30b}"
PROCESSOR="${PROCESSOR:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8}"
SUFFIX="${SUFFIX:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/benchmarks/guidellm${SUFFIX}}"
MAX_SECONDS="${MAX_SECONDS:-120}"
WARMUP="${WARMUP:-0.1}"

# ─── Scenario definitions ────────────────────────────────────────────
# Format: name|prompt_tokens|prompt_stdev|output_tokens|output_stdev|description
# Stdev adds realistic variance to request sizes (matching production traffic)
SCENARIOS=(
  "chat|2000|500|300|75|Multi-turn conversation (balanced, latency-sensitive)"
  "rag|8000|1500|256|64|RAG with 4-6 retrieved chunks (prefill-heavy)"
  "agentic|16000|3000|800|200|Tool-use agent: system prompt + tools + history + results"
  "codegen|4000|1000|1500|400|Code generation: file context to full functions"
  "summarization|12000|2000|300|75|Document summarization (pure prefill stress)"
)

# ─── Helpers ─────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(timestamp)] $*"; }

check_server() {
  log "Checking server at ${TARGET} ..."
  if ! curl -sf --max-time 5 "${TARGET}/health" > /dev/null 2>&1; then
    echo "ERROR: Server at ${TARGET} is not healthy. Start it first."
    exit 1
  fi
  log "Server is healthy."
}

run_scenario() {
  local name="$1" prompt_tokens="$2" prompt_stdev="$3"
  local output_tokens="$4" output_stdev="$5" description="$6"
  local scenario_dir="${OUTPUT_DIR}/${name}"
  mkdir -p "${scenario_dir}"

  # Compute reasonable min/max from mean +/- 2*stdev
  local prompt_min=$((prompt_tokens - 2 * prompt_stdev))
  local prompt_max=$((prompt_tokens + 2 * prompt_stdev))
  local output_min=$((output_tokens - 2 * output_stdev))
  local output_max=$((output_tokens + 2 * output_stdev))
  [[ $prompt_min -lt 1 ]] && prompt_min=1
  [[ $output_min -lt 1 ]] && output_min=1

  local data_spec="prompt_tokens=${prompt_tokens}"
  data_spec+=",prompt_tokens_stdev=${prompt_stdev}"
  data_spec+=",prompt_tokens_min=${prompt_min}"
  data_spec+=",prompt_tokens_max=${prompt_max}"
  data_spec+=",output_tokens=${output_tokens}"
  data_spec+=",output_tokens_stdev=${output_stdev}"
  data_spec+=",output_tokens_min=${output_min}"
  data_spec+=",output_tokens_max=${output_max}"

  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "Scenario: ${name}"
  log "  ${description}"
  log "  Prompt: ${prompt_tokens} +/- ${prompt_stdev} tokens"
  log "  Output: ${output_tokens} +/- ${output_stdev} tokens"
  log "  Profile: sweep (auto-ramp to saturation)"
  log "  Max seconds per rate: ${MAX_SECONDS}"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  guidellm benchmark \
    --target "${TARGET}" \
    --model "${MODEL}" \
    --processor "${PROCESSOR}" \
    --profile sweep \
    --data "${data_spec}" \
    --max-seconds "${MAX_SECONDS}" \
    --warmup "${WARMUP}" \
    --output-dir "${scenario_dir}" \
    --outputs json,csv \
    2>&1 | tee "${scenario_dir}/console.log"

  log "Results saved to ${scenario_dir}/"
  echo ""
}

# ─── Main ────────────────────────────────────────────────────────────
main() {
  echo ""
  log "╔══════════════════════════════════════════════════════════════╗"
  log "║  GuideLLM Parallel Benchmark — Nemotron on vLLM            ║"
  log "║  GPU: RTX PRO 6000 (96 GB)                                 ║"
  log "║  Model: ${MODEL}"
  log "╚══════════════════════════════════════════════════════════════╝"
  echo ""

  check_server
  mkdir -p "${OUTPUT_DIR}"

  # Filter scenarios if arguments provided
  local selected=("$@")

  for entry in "${SCENARIOS[@]}"; do
    IFS='|' read -r name prompt_tokens prompt_stdev output_tokens output_stdev description <<< "${entry}"

    # If specific scenarios requested, skip others
    if [[ ${#selected[@]} -gt 0 ]]; then
      local found=false
      for s in "${selected[@]}"; do
        if [[ "$s" == "$name" ]]; then found=true; break; fi
      done
      if [[ "$found" == "false" ]]; then continue; fi
    fi

    run_scenario "$name" "$prompt_tokens" "$prompt_stdev" "$output_tokens" "$output_stdev" "$description"
  done

  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "All benchmarks complete. Results in: ${OUTPUT_DIR}/"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

main "$@"
