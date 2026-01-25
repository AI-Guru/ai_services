#!/bin/bash

# Download GLM-4.7-Flash GGUF models using HuggingFace CLI
# Downloads Q4_K_XL (~18GB) and Q8_K_XL (~35GB) if not already present

set -e

echo "=========================================="
echo "GLM-4.7-Flash GGUF Model Download"
echo "=========================================="
echo ""
echo "Model: unsloth/GLM-4.7-Flash-GGUF"
echo "Downloading both quantizations:"
echo "  - Q4_K_XL: ~18GB (4-bit, fast)"
echo "  - Q8_K_XL: ~35GB (8-bit, higher quality)"
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

# Check and download Q4_K_XL
if [ -f "models/GLM-4.7-Flash-UD-Q4_K_XL.gguf" ]; then
    echo "✓ Q4_K_XL already downloaded (skipping)"
else
    echo "Downloading Q4_K_XL (~18GB)..."
    hf download unsloth/GLM-4.7-Flash-GGUF \
        --include "*UD-Q4_K_XL*" \
        --local-dir models/
    echo "✓ Q4_K_XL download complete"
fi

echo ""

# Check and download Q8_K_XL
if [ -f "models/GLM-4.7-Flash-UD-Q8_K_XL.gguf" ]; then
    echo "✓ Q8_K_XL already downloaded (skipping)"
else
    echo "Downloading Q8_K_XL (~35GB)..."
    echo "(This will take longer due to larger size)"
    hf download unsloth/GLM-4.7-Flash-GGUF \
        --include "*UD-Q8_K_XL*" \
        --local-dir models/
    echo "✓ Q8_K_XL download complete"
fi

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Available models:"
ls -lh models/GLM-4.7-Flash-UD-*K_XL.gguf 2>/dev/null || echo "No models found"
echo ""
echo "To use Q4_K_XL: Default in start_server.sh"
echo "To use Q8_K_XL: Edit start_server.sh and change MODEL_PATH"
echo ""
