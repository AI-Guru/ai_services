#!/bin/bash

PORT="${PORT:-11435}"
HOST="${HOST:-localhost}"
URL="http://$HOST:$PORT/v1/chat/completions"

echo "=== Long Poem Generation Benchmark ==="
echo "Server: $URL"
echo ""

START=$(date +%s%N)

RESPONSE=$(curl -s "$URL" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": "Write a very long, detailed epic poem (at least 2000 words) about the rise and fall of a civilization that discovered artificial intelligence. Include vivid imagery, metaphors, and multiple chapters. Do not stop until you have written at least 2000 words."
            }
        ],
        "max_tokens": 8192,
        "temperature": 0.7,
        "stream": false
    }')

END=$(date +%s%N)

DURATION_MS=$(( (END - START) / 1000000 ))
DURATION_S=$(echo "scale=2; $DURATION_MS / 1000" | bc)

CONTENT=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    content = data['choices'][0]['message']['content']
    usage = data.get('usage', {})
    prompt_tokens = usage.get('prompt_tokens', 'N/A')
    completion_tokens = usage.get('completion_tokens', 'N/A')
    total_tokens = usage.get('total_tokens', 'N/A')
    print(f'PROMPT_TOKENS={prompt_tokens}')
    print(f'COMPLETION_TOKENS={completion_tokens}')
    print(f'TOTAL_TOKENS={total_tokens}')
    word_count = len(content.split())
    print(f'WORD_COUNT={word_count}')
    print('---CONTENT---')
    print(content)
except Exception as e:
    print(f'ERROR={e}', file=sys.stderr)
    print(sys.stdin.read() if hasattr(sys.stdin, 'read') else '', file=sys.stderr)
    sys.exit(1)
" 2>&1)

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to parse response"
    echo "$RESPONSE" | head -200
    exit 1
fi

PROMPT_TOKENS=$(echo "$CONTENT" | grep "^PROMPT_TOKENS=" | cut -d= -f2)
COMPLETION_TOKENS=$(echo "$CONTENT" | grep "^COMPLETION_TOKENS=" | cut -d= -f2)
TOTAL_TOKENS=$(echo "$CONTENT" | grep "^TOTAL_TOKENS=" | cut -d= -f2)
WORD_COUNT=$(echo "$CONTENT" | grep "^WORD_COUNT=" | cut -d= -f2)
POEM=$(echo "$CONTENT" | sed -n '/^---CONTENT---$/,$ p' | tail -n +2)

if [ "$COMPLETION_TOKENS" != "N/A" ] && [ -n "$COMPLETION_TOKENS" ]; then
    TOKENS_PER_SEC=$(echo "scale=2; $COMPLETION_TOKENS / ($DURATION_MS / 1000)" | bc)
else
    TOKENS_PER_SEC="N/A"
fi

echo "=== Results ==="
echo "Duration:          ${DURATION_S}s"
echo "Prompt tokens:     $PROMPT_TOKENS"
echo "Completion tokens: $COMPLETION_TOKENS"
echo "Total tokens:      $TOTAL_TOKENS"
echo "Words generated:   $WORD_COUNT"
echo "Tokens/sec:        $TOKENS_PER_SEC"
echo ""
echo "=== Poem Preview (first 20 lines) ==="
echo "$POEM" | head -20
echo ""
echo "..."
