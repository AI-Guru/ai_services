#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"

case "${1:-qwen}" in
    qwen)
        HF_REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
        MODEL_NAME="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
        ;;
    qwopus)
        HF_REPO="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
        MODEL_NAME="Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Q4_K_M.gguf"
        ;;
    *)
        echo "Usage: $0 {qwen|qwopus}"
        echo "  qwen    Qwen3.5-35B-A3B (default)"
        echo "  qwopus  Qwen3.5-27B Claude Opus distilled"
        exit 1
        ;;
esac

mkdir -p "$MODEL_DIR"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists: $MODEL_PATH"
    exit 0
fi

echo "Downloading $MODEL_NAME from $HF_REPO..."
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='$HF_REPO',
    filename='$MODEL_NAME',
    local_dir='$MODEL_DIR'
)
print('Download complete: $MODEL_PATH')
"
