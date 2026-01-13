#!/bin/bash
set -e

BASE_URL="${VLLM_BASE_URL:-http://localhost:11435}"
MODEL="${VLLM_MODEL:-openai/gpt-oss-20b}"

echo "Testing vLLM server at ${BASE_URL}"
echo "Model: ${MODEL}"
echo ""

# Test 1: Health check
echo "Test 1: Health check"
if curl -s "${BASE_URL}/health" > /dev/null; then
    echo "  ✓ Server is healthy"
else
    echo "  ✗ Server health check failed"
    exit 1
fi
echo ""

# Test 2: List models
echo "Test 2: List models"
MODELS=$(curl -s "${BASE_URL}/v1/models" | jq -r '.data[].id' 2>/dev/null)
if [ -n "${MODELS}" ]; then
    echo "  ✓ Available models:"
    echo "${MODELS}" | sed 's/^/    - /'
else
    echo "  ✗ Failed to list models"
    exit 1
fi
echo ""

# Test 3: Long poem generation with timing
echo "Test 3: Long poem generation (measuring tokens/sec)"
echo "  Sending request..."
START_TIME=$(date +%s.%N)

RESPONSE=$(curl -s "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Write a very, very, very long poem about artificial intelligence, technology, and the future. Make it at least 500 words with multiple stanzas.\"}],
        \"temperature\": 0.7
    }")

END_TIME=$(date +%s.%N)
DURATION=$(echo "${END_TIME} - ${START_TIME}" | bc)

# Parse response
CONTENT=$(echo "${RESPONSE}" | jq -r '.choices[0].message.content // .choices[0].message.reasoning_content // empty' 2>/dev/null)
PROMPT_TOKENS=$(echo "${RESPONSE}" | jq -r '.usage.prompt_tokens // 0' 2>/dev/null)
COMPLETION_TOKENS=$(echo "${RESPONSE}" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
TOTAL_TOKENS=$(echo "${RESPONSE}" | jq -r '.usage.total_tokens // 0' 2>/dev/null)

if [ -z "${CONTENT}" ] || [ "${CONTENT}" = "null" ]; then
    echo "  ✗ Failed to generate content"
    echo "  Response: ${RESPONSE}"
    exit 1
fi

# Calculate tokens per second
TOKENS_PER_SEC=$(echo "scale=2; ${COMPLETION_TOKENS} / ${DURATION}" | bc)

echo "  ✓ Generation complete!"
echo ""
echo "  Metrics:"
echo "    - Duration: ${DURATION}s"
echo "    - Prompt tokens: ${PROMPT_TOKENS}"
echo "    - Completion tokens: ${COMPLETION_TOKENS}"
echo "    - Total tokens: ${TOTAL_TOKENS}"
echo "    - Tokens/second: ${TOKENS_PER_SEC} tok/s"
echo ""
echo "  Generated content (first 500 chars):"
echo "${CONTENT}" | head -c 500
echo ""
if [ ${#CONTENT} -gt 500 ]; then
    echo "  ... (truncated, total length: ${#CONTENT} chars)"
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ All tests passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Performance: ${TOKENS_PER_SEC} tokens/second"
