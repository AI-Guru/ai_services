#!/bin/bash

# Download GLM-4.7-Flash GGUF model using HuggingFace CLI
# Downloads BF16 (~60GB) if not already present

set -e

echo "=========================================="
echo "GLM-4.7-Flash GGUF Model Download"
echo "=========================================="
echo ""
echo "Model: unsloth/GLM-4.7-Flash-GGUF"
echo "Downloading the following quantization:"
echo "  - BF16:    ~60GB (16-bit, higher quality)"
echo "Local directory: models/"
echo ""

# Check if hf command exists
if ! command -v hf &> /dev/null; then
    echo "ERROR: 'hf' command not found!"
    echo "Please install huggingface-cli"
    exit 1
fi

# Create models directory if it doesn't exist
mkdir -p models

# Check and download BF16
if [ -f "models/GLM-4.7-Flash-BF16-00001-of-00002.gguf" ]; then
    echo "✓ BF16 already downloaded (skipping)"
else
    echo "Downloading BF16 (~60GB)..."
    echo "(This will take much longer due to larger size)"
    hf download unsloth/GLM-4.7-Flash-GGUF \
        --include "*BF16*" \
        --local-dir models/
    echo "✓ BF16 download complete"
fi

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Available models:"
ls -lh models/GLM-4.7-Flash-*.gguf models/BF16/GLM-4.7-Flash-BF16-0000*-of-00002.gguf 2>/dev/null || echo "No models found"
echo ""
echo "To use BF16: Run start_server_bf16.sh"
echo ""
