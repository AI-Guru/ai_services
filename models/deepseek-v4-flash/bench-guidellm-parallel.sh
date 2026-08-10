#!/bin/bash
# GuideLLM parallel benchmarking for DeepSeek-V4-Flash-0731 (284B/13B MoE, TP2)
# Measures concurrency scaling across workload profiles on 2x RTX PRO 6000 (SM120).
#
# Token counts match the family convention (see ../qwen3.6/bench-guidellm-parallel.sh)
# so results are directly comparable across model families.
#
# Usage:
#   CONFIG=256k ./bench-guidellm-parallel.sh              # all scenarios
#   CONFIG=1m   ./bench-guidellm-parallel.sh chat rag     # specific scenarios
#
# Bring the matching server up FIRST — the two configs are mutually exclusive:
#   docker compose -f docker-compose.vllm-284b-mxfp4-256k-rtx.yml up -d
#   docker compose -f docker-compose.vllm-284b-mxfp4-1m-rtx.yml   up -d
#
# CONFIG only labels the output directory; it does not start or reconfigure the
# server. Setting it wrong silently mislabels results, so check /v1/models first.

set -euo pipefail

TARGET="${TARGET:-http://localhost:11437}"
MODEL="${MODEL:-deepseek-v4-flash}"
PROCESSOR="${PROCESSOR:-deepseek-ai/DeepSeek-V4-Flash-0731}"
CONFIG="${CONFIG:-unknown}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/benchmarks/guidellm/${CONFIG}}"
MAX_SECONDS="${MAX_SECONDS:-120}"
WARMUP="${WARMUP:-0.1}"
GUIDELLM="${GUIDELLM:-/opt/guidellm/bin/guidellm}"

# ─── Scenario definitions ────────────────────────────────────────────
# Format: name|prompt_tokens|prompt_stdev|output_tokens|output_stdev|description
SCENARIOS=(
  "chat|2000|500|300|75|Multi-turn conversation (balanced, latency-sensitive)"
  "rag|8000|1500|256|64|RAG with 4-6 retrieved chunks (prefill-heavy)"
  "agentic|16000|3000|800|200|Tool-use agent: system prompt + tools + history + results"
  "codegen|4000|1000|1500|400|Code generation: file context to full functions"
  "summarization|12000|2000|300|75|Document summarization (pure prefill stress)"
)

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

check_server() {
  log "Checking server at ${TARGET} ..."
  if ! curl -sf --max-time 5 "${TARGET}/health" > /dev/null 2>&1; then
    echo "ERROR: Server at ${TARGET} is not healthy. Start it first."
    exit 1
  fi
  local served_len
  served_len=$(curl -sf --max-time 5 "${TARGET}/v1/models" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['max_model_len'])" 2>/dev/null || echo "?")
  log "Server healthy. Serving max_model_len=${served_len}, labelling results CONFIG=${CONFIG}."
}

run_scenario() {
  local name="$1" prompt_tokens="$2" prompt_stdev="$3"
  local output_tokens="$4" output_stdev="$5" description="$6"
  local scenario_dir="${OUTPUT_DIR}/${name}"
  mkdir -p "${scenario_dir}"

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
  log "Scenario: ${name}  [CONFIG=${CONFIG}]"
  log "  ${description}"
  log "  Prompt: ${prompt_tokens} +/- ${prompt_stdev} tokens"
  log "  Output: ${output_tokens} +/- ${output_stdev} tokens"
  log "  Profile: sweep (auto-ramp to saturation), ${MAX_SECONDS}s per rate"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # NOTE 1: output is REDIRECTED, not piped through tee. guidellm deadlocks on
  # SIGPIPE when its stdout is a pipe that closes early.
  # NOTE 2: --outputs takes explicit FILENAMES here, not the documented "json,csv"
  # aliases. On guidellm 0.5.4 the alias form resolves the output path to the bare
  # directory and dies at the end of an otherwise-successful run with
  # "ValueError: Unsupported file type:  for <dir>". Verified 2026-08-10.
  "${GUIDELLM}" benchmark \
    --target "${TARGET}" \
    --model "${MODEL}" \
    --processor "${PROCESSOR}" \
    --profile sweep \
    --data "${data_spec}" \
    --max-seconds "${MAX_SECONDS}" \
    --warmup "${WARMUP}" \
    --output-dir "${scenario_dir}" \
    --outputs benchmark.json,benchmark.csv \
    > "${scenario_dir}/console.log" 2>&1 \
    || log "WARNING: scenario ${name} exited non-zero — see ${scenario_dir}/console.log"

  log "Results saved to ${scenario_dir}/"
  echo ""
}

main() {
  echo ""
  log "╔══════════════════════════════════════════════════════════════╗"
  log "║  GuideLLM sweep — DeepSeek-V4-Flash-0731 (284B/13B MoE)      ║"
  log "║  2x RTX PRO 6000 Blackwell (SM120), TP2 + EP, PCIe           ║"
  log "╚══════════════════════════════════════════════════════════════╝"
  echo ""

  check_server
  mkdir -p "${OUTPUT_DIR}"

  local selected=("$@")

  for entry in "${SCENARIOS[@]}"; do
    IFS='|' read -r name prompt_tokens prompt_stdev output_tokens output_stdev description <<< "${entry}"

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
  log "Sweep complete for CONFIG=${CONFIG}. Results in: ${OUTPUT_DIR}/"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

main "$@"
