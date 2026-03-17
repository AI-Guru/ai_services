#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENCODINGS_DIR="${SCRIPT_DIR}/encodings"

echo "Preparing vLLM gpt-oss environment..."

# Create encodings directory
echo "Creating encodings directory..."
mkdir -p "${ENCODINGS_DIR}"

# Download tiktoken encoding files
echo "Downloading tiktoken encoding files..."

if [ ! -f "${ENCODINGS_DIR}/o200k_base.tiktoken" ]; then
    echo "  - Downloading o200k_base.tiktoken..."
    curl -sL -o "${ENCODINGS_DIR}/o200k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    echo "    ✓ Downloaded o200k_base.tiktoken ($(du -h "${ENCODINGS_DIR}/o200k_base.tiktoken" | cut -f1))"
else
    echo "  ✓ o200k_base.tiktoken already exists"
fi

if [ ! -f "${ENCODINGS_DIR}/cl100k_base.tiktoken" ]; then
    echo "  - Downloading cl100k_base.tiktoken..."
    curl -sL -o "${ENCODINGS_DIR}/cl100k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    echo "    ✓ Downloaded cl100k_base.tiktoken ($(du -h "${ENCODINGS_DIR}/cl100k_base.tiktoken" | cut -f1))"
else
    echo "  ✓ cl100k_base.tiktoken already exists"
fi

echo ""
echo "✓ Preparation complete!"
echo ""
echo "You can now start the vLLM server with:"
echo "  docker compose up -d"
