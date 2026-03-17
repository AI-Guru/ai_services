#!/bin/bash
# Parallel stress test for Ollama and vLLM
# Tests concurrent request handling using xargs for parallelism

set -e

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
VLLM_URL="${VLLM_URL:-http://localhost:11435}"
NUM_REQUESTS="${NUM_REQUESTS:-10}"
CONCURRENCY="${CONCURRENCY:-5}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Parallel Stress Test: gpt-oss-20b"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  Total requests: ${NUM_REQUESTS}"
echo "  Concurrency: ${CONCURRENCY}"
echo "  Ollama URL: ${OLLAMA_URL}"
echo "  vLLM URL: ${VLLM_URL}"
echo ""

# Check availability
echo "Checking server availability..."
echo ""

OLLAMA_AVAILABLE=false
VLLM_AVAILABLE=false

if curl -s --max-time 5 "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo "  ✓ Ollama available"
    OLLAMA_AVAILABLE=true
else
    echo "  ✗ Ollama not responding"
fi

if curl -s --max-time 5 "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "  ✓ vLLM available"
    VLLM_AVAILABLE=true
else
    echo "  ✗ vLLM not responding"
fi

echo ""

if [ "${OLLAMA_AVAILABLE}" = "false" ] && [ "${VLLM_AVAILABLE}" = "false" ]; then
    echo "Error: Neither server is available"
    exit 1
fi

# Test vLLM
if [ "${VLLM_AVAILABLE}" = "true" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing vLLM (${VLLM_URL})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Sending ${NUM_REQUESTS} requests with ${CONCURRENCY} parallel workers..."

    START=$(date +%s.%N)

    seq 1 ${NUM_REQUESTS} | xargs -P ${CONCURRENCY} -I {} bash -c "
        curl -s '${VLLM_URL}/v1/chat/completions' \
            -H 'Content-Type: application/json' \
            -d '{
                \"model\": \"openai/gpt-oss-20b\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Write a short poem about artificial intelligence and its impact on society\"}],
                \"temperature\": 0.7,
                \"max_tokens\": 100
            }' > /dev/null
        echo -n '.'
    "

    END=$(date +%s.%N)
    VLLM_DURATION=$(echo "${END} - ${START}" | bc)
    VLLM_RPS=$(echo "scale=2; ${NUM_REQUESTS} / ${VLLM_DURATION}" | bc)

    echo ""
    echo ""
    echo "  ✓ Completed"
    echo "  Duration: ${VLLM_DURATION}s"
    echo "  Throughput: ${VLLM_RPS} requests/sec"
    echo ""
fi

# Test Ollama
if [ "${OLLAMA_AVAILABLE}" = "true" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing Ollama (${OLLAMA_URL})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Sending ${NUM_REQUESTS} requests with ${CONCURRENCY} parallel workers..."

    START=$(date +%s.%N)

    seq 1 ${NUM_REQUESTS} | xargs -P ${CONCURRENCY} -I {} bash -c "
        curl -s '${OLLAMA_URL}/api/generate' \
            -d '{
                \"model\": \"gpt-oss:20b\",
                \"prompt\": \"Write a short poem about artificial intelligence and its impact on society\",
                \"stream\": false,
                \"options\": {\"temperature\": 0.7, \"num_predict\": 100}
            }' > /dev/null
        echo -n '.'
    "

    END=$(date +%s.%N)
    OLLAMA_DURATION=$(echo "${END} - ${START}" | bc)
    OLLAMA_RPS=$(echo "scale=2; ${NUM_REQUESTS} / ${OLLAMA_DURATION}" | bc)

    echo ""
    echo ""
    echo "  ✓ Completed"
    echo "  Duration: ${OLLAMA_DURATION}s"
    echo "  Throughput: ${OLLAMA_RPS} requests/sec"
    echo ""
fi

# Comparison
if [ "${VLLM_AVAILABLE}" = "true" ] && [ "${OLLAMA_AVAILABLE}" = "true" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Comparison Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    printf "%-25s %15s %15s\n" "Metric" "vLLM" "Ollama"
    echo "─────────────────────────────────────────────────────"
    printf "%-25s %15s %15s\n" "Duration (s)" "${VLLM_DURATION}" "${OLLAMA_DURATION}"
    printf "%-25s %15s %15s\n" "Requests/sec" "${VLLM_RPS}" "${OLLAMA_RPS}"
    echo ""

    SPEEDUP=$(echo "scale=2; ${OLLAMA_DURATION} / ${VLLM_DURATION}" | bc)
    echo "Result: vLLM is ${SPEEDUP}x faster for concurrent requests"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Recommended Test Progression:"
echo ""
echo "  # Phase 1: Baseline (verify setup)"
echo "  ./stresstest.sh"
echo ""
echo "  # Phase 2: Double concurrency"
echo "  NUM_REQUESTS=20 CONCURRENCY=10 ./stresstest.sh"
echo ""
echo "  # Phase 3: LangGraph typical workload (16 concurrent)"
echo "  NUM_REQUESTS=50 CONCURRENCY=16 ./stresstest.sh"
echo ""
echo "  # Phase 4: Production load (25 concurrent)"
echo "  NUM_REQUESTS=100 CONCURRENCY=25 ./stresstest.sh"
echo ""
echo "  # Phase 5: High concurrency (50 concurrent)"
echo "  NUM_REQUESTS=200 CONCURRENCY=50 ./stresstest.sh"
echo ""
echo "  # Phase 6: Maximum throughput (100 concurrent)"
echo "  NUM_REQUESTS=500 CONCURRENCY=100 ./stresstest.sh"
echo ""
echo "Monitor GPU usage: watch -n 1 nvidia-smi"
echo ""
