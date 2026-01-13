#!/bin/bash
set -e

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
VLLM_URL="${VLLM_URL:-http://localhost:11435}"
PROMPT="${TEST_PROMPT:-Write a very, very, very long poem about artificial intelligence, technology, and the future. Make it at least 500 words with multiple stanzas.}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Comparing gpt-oss-20b: Ollama vs vLLM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Ollama: ${OLLAMA_URL}"
echo "vLLM:   ${VLLM_URL}"
echo ""

# Check availability of both servers first
echo "Checking server availability..."
echo ""

# Check Ollama
echo -n "Ollama (port 11434): "
if curl -s --max-time 5 "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo "✓ Available"
    OLLAMA_AVAILABLE=true
else
    echo "✗ Not responding"
    OLLAMA_AVAILABLE=false
fi

# Check vLLM
echo -n "vLLM (port 11435):   "
if curl -s --max-time 5 "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "✓ Available"
    VLLM_AVAILABLE=true
else
    echo "✗ Not responding"
    VLLM_AVAILABLE=false
fi

echo ""

# Exit if neither is available
if [ "${OLLAMA_AVAILABLE}" = "false" ] && [ "${VLLM_AVAILABLE}" = "false" ]; then
    echo "Error: Neither Ollama nor vLLM servers are responding"
    exit 1
fi

# Test Ollama
if [ "${OLLAMA_AVAILABLE}" = "true" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing Ollama (port 11434)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Skipping Ollama (not available)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

if [ "${OLLAMA_AVAILABLE}" = "true" ]; then
    echo "  Sending request..."
    OLLAMA_START=$(date +%s.%N)

    OLLAMA_RESPONSE=$(curl -s "${OLLAMA_URL}/api/generate" \
        -d "{
            \"model\": \"gpt-oss:20b\",
            \"prompt\": \"${PROMPT}\",
            \"stream\": false,
            \"options\": {
                \"temperature\": 0.7
            }
        }")

    OLLAMA_END=$(date +%s.%N)
    OLLAMA_DURATION=$(echo "${OLLAMA_END} - ${OLLAMA_START}" | bc)

    # Parse Ollama response
    OLLAMA_CONTENT=$(echo "${OLLAMA_RESPONSE}" | jq -r '.response // empty' 2>/dev/null)
    OLLAMA_PROMPT_TOKENS=$(echo "${OLLAMA_RESPONSE}" | jq -r '.prompt_eval_count // 0' 2>/dev/null)
    OLLAMA_COMPLETION_TOKENS=$(echo "${OLLAMA_RESPONSE}" | jq -r '.eval_count // 0' 2>/dev/null)
    OLLAMA_TOTAL_TOKENS=$((OLLAMA_PROMPT_TOKENS + OLLAMA_COMPLETION_TOKENS))

    if [ -z "${OLLAMA_CONTENT}" ] || [ "${OLLAMA_CONTENT}" = "null" ]; then
        echo "  ✗ Failed to generate content"
        OLLAMA_SUCCESS=false
    else
        OLLAMA_TOKENS_PER_SEC=$(echo "scale=2; ${OLLAMA_COMPLETION_TOKENS} / ${OLLAMA_DURATION}" | bc)

        echo "  ✓ Generation complete!"
        echo ""
        echo "  Metrics:"
        echo "    - Duration: ${OLLAMA_DURATION}s"
        echo "    - Prompt tokens: ${OLLAMA_PROMPT_TOKENS}"
        echo "    - Completion tokens: ${OLLAMA_COMPLETION_TOKENS}"
        echo "    - Total tokens: ${OLLAMA_TOTAL_TOKENS}"
        echo "    - Tokens/second: ${OLLAMA_TOKENS_PER_SEC} tok/s"
        echo ""
        echo "  Generated content (first 300 chars):"
        echo "${OLLAMA_CONTENT}" | head -c 300
        echo ""
        echo "  ... (total length: ${#OLLAMA_CONTENT} chars)"
        OLLAMA_SUCCESS=true
    fi
fi

echo ""

# Test vLLM
if [ "${VLLM_AVAILABLE}" = "true" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing vLLM (port 11435)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Skipping vLLM (not available)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

if [ "${VLLM_AVAILABLE}" = "true" ]; then
    echo "  Sending request..."
    VLLM_START=$(date +%s.%N)

    VLLM_RESPONSE=$(curl -s "${VLLM_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"openai/gpt-oss-20b\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
            \"temperature\": 0.7
        }")

    VLLM_END=$(date +%s.%N)
    VLLM_DURATION=$(echo "${VLLM_END} - ${VLLM_START}" | bc)

    # Parse vLLM response
    VLLM_CONTENT=$(echo "${VLLM_RESPONSE}" | jq -r '.choices[0].message.content // .choices[0].message.reasoning_content // empty' 2>/dev/null)
    VLLM_PROMPT_TOKENS=$(echo "${VLLM_RESPONSE}" | jq -r '.usage.prompt_tokens // 0' 2>/dev/null)
    VLLM_COMPLETION_TOKENS=$(echo "${VLLM_RESPONSE}" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
    VLLM_TOTAL_TOKENS=$(echo "${VLLM_RESPONSE}" | jq -r '.usage.total_tokens // 0' 2>/dev/null)

    if [ -z "${VLLM_CONTENT}" ] || [ "${VLLM_CONTENT}" = "null" ]; then
        echo "  ✗ Failed to generate content"
        VLLM_SUCCESS=false
    else
        VLLM_TOKENS_PER_SEC=$(echo "scale=2; ${VLLM_COMPLETION_TOKENS} / ${VLLM_DURATION}" | bc)

        echo "  ✓ Generation complete!"
        echo ""
        echo "  Metrics:"
        echo "    - Duration: ${VLLM_DURATION}s"
        echo "    - Prompt tokens: ${VLLM_PROMPT_TOKENS}"
        echo "    - Completion tokens: ${VLLM_COMPLETION_TOKENS}"
        echo "    - Total tokens: ${VLLM_TOTAL_TOKENS}"
        echo "    - Tokens/second: ${VLLM_TOKENS_PER_SEC} tok/s"
        echo ""
        echo "  Generated content (first 300 chars):"
        echo "${VLLM_CONTENT}" | head -c 300
        echo ""
        echo "  ... (total length: ${#VLLM_CONTENT} chars)"
        VLLM_SUCCESS=true
    fi
fi

echo ""

# Comparison
if [ "${OLLAMA_SUCCESS}" = "true" ] && [ "${VLLM_SUCCESS}" = "true" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Comparison Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    printf "%-25s %15s %15s\n" "Metric" "Ollama" "vLLM"
    echo "─────────────────────────────────────────────────────"
    printf "%-25s %15s %15s\n" "Duration (s)" "${OLLAMA_DURATION}" "${VLLM_DURATION}"
    printf "%-25s %15s %15s\n" "Completion tokens" "${OLLAMA_COMPLETION_TOKENS}" "${VLLM_COMPLETION_TOKENS}"
    printf "%-25s %15s %15s\n" "Tokens/second" "${OLLAMA_TOKENS_PER_SEC}" "${VLLM_TOKENS_PER_SEC}"
    printf "%-25s %15s %15s\n" "Output length (chars)" "${#OLLAMA_CONTENT}" "${#VLLM_CONTENT}"
    echo ""

    # Calculate speedup
    SPEEDUP=$(echo "scale=2; ${VLLM_TOKENS_PER_SEC} / ${OLLAMA_TOKENS_PER_SEC}" | bc)

    if (( $(echo "${VLLM_TOKENS_PER_SEC} > ${OLLAMA_TOKENS_PER_SEC}" | bc -l) )); then
        echo "Result: vLLM is ${SPEEDUP}x faster than Ollama"
    elif (( $(echo "${OLLAMA_TOKENS_PER_SEC} > ${VLLM_TOKENS_PER_SEC}" | bc -l) )); then
        SPEEDUP=$(echo "scale=2; ${OLLAMA_TOKENS_PER_SEC} / ${VLLM_TOKENS_PER_SEC}" | bc)
        echo "Result: Ollama is ${SPEEDUP}x faster than vLLM"
    else
        echo "Result: Performance is approximately equal"
    fi
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
