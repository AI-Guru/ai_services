#!/bin/bash

# Test GLM-4.7-Flash server with curl
# Tests basic chat and displays VRAM usage

SERVER_URL="http://127.0.0.1:11346"

echo "=========================================="
echo "GLM-4.7-Flash Server Test"
echo "=========================================="
echo ""

# Test 1: Basic Chat
echo "Test 1: Basic Chat Completion"
echo "Query: What is 2+2? Answer briefly."
echo ""
echo "Response:"

curl -s ${SERVER_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/GLM-4.7-Flash",
    "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
    "temperature": 0.7,
    "max_tokens": 100
  }' | jq -r '.choices[0].message.content'

echo ""
echo "=========================================="
echo ""

# Test 2: Tool Calling Example
echo "Test 2: Tool Calling"
echo "Query: What's the weather like in Tokyo?"
echo ""
echo "Response:"

curl -s ${SERVER_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/GLM-4.7-Flash",
    "messages": [{"role": "user", "content": "What is the weather like in Tokyo?"}],
    "temperature": 0.7,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit"
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  }' | jq '.'

echo ""
echo "=========================================="
echo ""

# Check VRAM usage
echo "VRAM Usage:"
echo ""
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader | while IFS=', ' read -r name used total util; do
    percentage=$(awk "BEGIN {printf \"%.1f\", ($used / $total) * 100}")
    echo "GPU: $name"
    echo "VRAM: ${used} MB / ${total} MB (${percentage}%)"
    echo "Utilization: ${util}%"
done

echo ""
echo "=========================================="
echo "Tests Complete"
echo "=========================================="
