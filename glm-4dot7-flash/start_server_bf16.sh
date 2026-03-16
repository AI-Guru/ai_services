#!/bin/bash

# GLM-4.7-Flash Server Startup Script (BF16 version)
# Configured for tool-calling with full 202K context window
# Port: 11346
# Uses 8-bit quantization for higher quality

set -e

MODEL_PATH="models/BF16/GLM-4.7-Flash-BF16-00001-of-00002.gguf"
SERVER_BIN="llama.cpp/build/bin/llama-server"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please run ./download_model.sh first"
    exit 1
fi

# Check if server binary exists
if [ ! -f "$SERVER_BIN" ]; then
    echo "ERROR: llama-server not found at $SERVER_BIN"
    echo "Please run ./build.sh first"
    exit 1
fi

echo "=========================================="
echo "GLM-4.7-Flash Server (BF16)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Quantization: 16-bit (higher quality)"
echo "Port: 11346"
echo "Context: 202,752 tokens (full)"
echo "Mode: Tool-calling optimized"
echo "=========================================="
echo ""
echo "Starting server..."
echo "API endpoint: http://0.0.0.0:11346/v1"
echo "Accessible from network at: http://hordak:11346/v1"
echo ""

./$SERVER_BIN \
    --model "$MODEL_PATH" \
    --alias "unsloth/GLM-4.7-Flash" \
    --host 0.0.0.0 \
    --threads -1 \
    --fit on \
    --temp 0.7 \
    --top-p 1.0 \
    --min-p 0.01 \
    --ctx-size 202752 \
    --port 11346 \
    --jinja \
    --repeat-penalty 1.0
