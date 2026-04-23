#!/bin/bash
#
# Qwen3.6 on Apple Silicon via MLX (35B MoE or 27B dense)
#
# Usage:
#   ./run-qwen-mlx.sh --install    # Create conda env and install packages
#   ./run-qwen-mlx.sh --uninstall  # Remove conda env
#   ./run-qwen-mlx.sh --upgrade    # Upgrade packages in conda env
#   ./run-qwen-mlx.sh --serve      # Start the OpenAI-compatible server
#
# Requirements: conda (Miniconda or Anaconda)

set -e

# =============================================================================
# Configuration
# =============================================================================

ENV_NAME="qwen36-mlx"
PYTHON_VERSION="3.11"
PORT="11436"
CONTEXT_LENGTH="262144"  # 256K tokens (model maximum)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_PATH="${SCRIPT_DIR}/qwen3.6_chat_template.jinja"

# =============================================================================
# Model selection — uncomment ONE model
# =============================================================================

# 35B MoE (3B active) — default, ~20 GB, requires 24+ GB unified memory
MODEL="mlx-community/Qwen3.6-35B-A3B-4bit"
# MODEL="mlx-community/Qwen3.6-35B-A3B-4bit-DWQ" # 35B MoE DWQ — same size, better quality
# MODEL="mlx-community/Qwen3.6-35B-A3B-8bit"     # 35B MoE 8-bit — ~40 GB, best MoE quality

# 27B dense — all 27B params active, stronger per-token but bandwidth-bound (~30–40 tok/s)
# No mlx-community checkpoint yet; Unsloth Dynamic Quant variants are the best available:
# MODEL="unsloth/Qwen3.6-27B-UD-MLX-4bit"        # ~16 GB, recommended for 24 GB Macs
# MODEL="unsloth/Qwen3.6-27B-UD-MLX-6bit"        # ~22 GB, better quality
# MODEL="unsloth/Qwen3.6-27B-MLX-8bit"           # ~35 GB, requires 48+ GB unified memory

# =============================================================================
# Helper functions
# =============================================================================

info() {
    echo "[INFO] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

check_conda() {
    if ! command -v conda &> /dev/null; then
        error "conda not found. Install Miniconda or Anaconda first:

  # Miniconda (recommended):
  brew install miniconda

  # Or download from:
  https://docs.conda.io/en/latest/miniconda.html"
    fi
}

env_exists() {
    conda env list | grep -q "^${ENV_NAME} "
}

activate_env() {
    # Source conda for script context
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
}

# =============================================================================
# Commands
# =============================================================================

cmd_install() {
    check_conda

    if env_exists; then
        info "Environment '${ENV_NAME}' already exists. Use --upgrade to update packages."
        exit 0
    fi

    info "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

    info "Installing mlx-openai-server..."
    activate_env
    pip install mlx-openai-server

    info "Installation complete!"
    info ""
    info "Next steps:"
    info "  ./run-qwen-mlx.sh --serve"
    info ""
    info "The model (~20GB) will download on first serve."
}

cmd_uninstall() {
    check_conda

    if ! env_exists; then
        info "Environment '${ENV_NAME}' does not exist."
        exit 0
    fi

    info "Removing conda environment '${ENV_NAME}'..."
    conda env remove -n "$ENV_NAME" -y

    info "Environment removed."
    info ""
    info "Note: Cached models remain in ~/.cache/huggingface"
    info "To remove them: rm -rf ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit"
}

cmd_upgrade() {
    check_conda

    if ! env_exists; then
        error "Environment '${ENV_NAME}' does not exist. Run --install first."
    fi

    info "Upgrading packages in '${ENV_NAME}'..."
    activate_env
    pip install --upgrade mlx-openai-server mlx-lm mlx

    info "Upgrade complete!"
}

cmd_serve() {
    check_conda

    if ! env_exists; then
        error "Environment '${ENV_NAME}' does not exist. Run --install first."
    fi

    if [[ ! -f "$TEMPLATE_PATH" ]]; then
        error "Chat template not found: ${TEMPLATE_PATH}"
    fi

    info "Starting Qwen3.6 server on port ${PORT}..."
    info "Model: ${MODEL}"
    info "Context: ${CONTEXT_LENGTH} tokens (256K)"
    info "Template: ${TEMPLATE_PATH}"
    info ""
    info "API endpoint: http://localhost:${PORT}/v1"
    info "Press Ctrl+C to stop"
    info ""

    activate_env
    exec mlx-openai-server launch \
        --model-path "$MODEL" \
        --model-type lm \
        --chat-template-file "$TEMPLATE_PATH" \
        --context-length "$CONTEXT_LENGTH" \
        --reasoning-parser qwen3_5 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --port "$PORT"
}

show_usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  --install    Create conda environment and install packages"
    echo "  --uninstall  Remove conda environment"
    echo "  --upgrade    Upgrade packages in conda environment"
    echo "  --serve      Start the OpenAI-compatible server"
    echo ""
    echo "Configuration (edit script to change):"
    echo "  Environment: ${ENV_NAME}"
    echo "  Model:       ${MODEL}"
    echo "  Port:        ${PORT}"
}

# =============================================================================
# Main
# =============================================================================

case "${1:-}" in
    --install)
        cmd_install
        ;;
    --uninstall)
        cmd_uninstall
        ;;
    --upgrade)
        cmd_upgrade
        ;;
    --serve)
        cmd_serve
        ;;
    --help|-h)
        show_usage
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
