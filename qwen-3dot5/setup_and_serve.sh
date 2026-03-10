#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR="$SCRIPT_DIR/llama.cpp"
MODEL_DIR="$SCRIPT_DIR/models"
TEMPLATE_FILE="$SCRIPT_DIR/qwen3.5_chat_template.jinja"

# --- Model selection ---
# MODEL=qwen (default)  Qwen3.5-35B-A3B  + patched jinja template for developer role
# MODEL=qwopus          Qwen3.5-27B with Claude Opus 4.6 reasoning distilled (no template patch needed)
MODEL="${MODEL:-qwen}"

case "$MODEL" in
    qwen)
        MODEL_NAME="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
        HF_REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
        NEEDS_TEMPLATE=true
        ;;
    qwopus)
        MODEL_NAME="Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Q4_K_M.gguf"
        HF_REPO="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
        NEEDS_TEMPLATE=false
        ;;
    *)
        echo "Unknown MODEL: $MODEL (use 'qwen' or 'qwopus')"
        exit 1
        ;;
esac

MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

CUDA_ARCH="${CUDA_ARCH:-86}"
NUM_JOBS="${NUM_JOBS:-$(nproc)}"

# Server defaults
CTX_SIZE="${CTX_SIZE:-262144}"
PORT="${PORT:-11435}"
HOST="${HOST:-0.0.0.0}"
GPU_LAYERS="${GPU_LAYERS:-99}"
PARALLEL="${PARALLEL:-1}"

build_llama_server() {
    echo "=== Building llama-server ==="

    if [ ! -d "$LLAMA_CPP_DIR" ]; then
        echo "Cloning llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
    else
        echo "Updating llama.cpp..."
        git -C "$LLAMA_CPP_DIR" pull origin master
    fi

    echo "Configuring with CUDA (arch=$CUDA_ARCH)..."
    cmake -B "$LLAMA_CPP_DIR/build" -S "$LLAMA_CPP_DIR" \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

    echo "Building llama-server with $NUM_JOBS jobs..."
    cmake --build "$LLAMA_CPP_DIR/build" --target llama-server -j"$NUM_JOBS"

    LLAMA_SERVER=$(find "$LLAMA_CPP_DIR/build" -name "llama-server" -type f -executable 2>/dev/null | head -1)
    if [ -z "$LLAMA_SERVER" ]; then
        echo "ERROR: llama-server binary not found after build"
        exit 1
    fi
    echo "Built: $LLAMA_SERVER"
    "$LLAMA_SERVER" --version
}

download_model() {
    echo "=== Downloading model ==="
    mkdir -p "$MODEL_DIR"

    if [ -f "$MODEL_PATH" ]; then
        echo "Model already exists: $MODEL_PATH"
        return 0
    fi

    echo "Downloading $MODEL_NAME from $HF_REPO..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='$HF_REPO',
    filename='$MODEL_NAME',
    local_dir='$MODEL_DIR'
)
print('Download complete.')
"
}

serve() {
    LLAMA_SERVER=$(find "$LLAMA_CPP_DIR/build" -name "llama-server" -type f -executable 2>/dev/null | head -1)
    if [ -z "$LLAMA_SERVER" ]; then
        echo "ERROR: llama-server not found. Run '$0 build' first."
        exit 1
    fi
    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: Model not found at $MODEL_PATH. Run '$0 download' first."
        exit 1
    fi

    echo "=== Starting llama-server ==="
    echo "Variant:  $MODEL"
    echo "Model:    $MODEL_PATH"
    echo "Context:  $CTX_SIZE"
    echo "Port:     $PORT"
    echo "GPU layers: $GPU_LAYERS"

    TEMPLATE_ARGS=()
    if [ "$NEEDS_TEMPLATE" = true ]; then
        if [ ! -f "$TEMPLATE_FILE" ]; then
            echo "ERROR: Template not found at $TEMPLATE_FILE"
            exit 1
        fi
        TEMPLATE_ARGS=(--chat-template-file "$TEMPLATE_FILE")
        echo "Template: $TEMPLATE_FILE"
    fi
    echo ""

    exec "$LLAMA_SERVER" \
        -m "$MODEL_PATH" \
        -ngl "$GPU_LAYERS" \
        -c "$CTX_SIZE" \
        -fa on \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        -np "$PARALLEL" \
        --host "$HOST" \
        --port "$PORT" \
        "${TEMPLATE_ARGS[@]}"
}

usage() {
    echo "Usage: $0 {build|download|serve|all}"
    echo ""
    echo "Commands:"
    echo "  build     Clone/update llama.cpp and build llama-server"
    echo "  download  Download the GGUF model from Hugging Face"
    echo "  serve     Start the llama-server"
    echo "  all       Build, download, and serve (default)"
    echo ""
    echo "Model selection (MODEL env var):"
    echo "  MODEL=qwen    Qwen3.5-35B-A3B + patched template (default)"
    echo "  MODEL=qwopus  Qwen3.5-27B Claude Opus reasoning distilled"
    echo ""
    echo "Examples:"
    echo "  MODEL=qwen   $0 all     # Run Qwen3.5-35B with template fix"
    echo "  MODEL=qwopus $0 all     # Run Qwopus distilled model"
    echo ""
    echo "Environment variables:"
    echo "  CUDA_ARCH   CUDA architecture (default: 86 for RTX 3090)"
    echo "  CTX_SIZE    Context size (default: 262144)"
    echo "  PORT        Server port (default: 11435)"
    echo "  GPU_LAYERS  GPU layers to offload (default: 99)"
    echo "  PARALLEL    Number of parallel sequences (default: 1)"
    echo "  NUM_JOBS    Build parallelism (default: nproc)"
}

case "${1:-all}" in
    build)
        build_llama_server
        ;;
    download)
        download_model
        ;;
    serve)
        serve
        ;;
    all)
        build_llama_server
        download_model
        serve
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac
