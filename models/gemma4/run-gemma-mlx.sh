#!/bin/bash
#
# Gemma 4 on Apple Silicon via MLX
#
# Usage:
#   ./run-gemma-mlx.sh --install    # Create conda env and install packages
#   ./run-gemma-mlx.sh --uninstall  # Remove conda env
#   ./run-gemma-mlx.sh --upgrade    # Upgrade packages in conda env
#   ./run-gemma-mlx.sh --serve      # Start the OpenAI-compatible server
#
# Requirements: conda (Miniconda or Anaconda)

set -e

# =============================================================================
# Configuration
# =============================================================================

ENV_NAME="gemma-mlx"
PYTHON_VERSION="3.11"
CONTEXT_LENGTH="131072"  # 128K tokens

# =============================================================================
# Model selection — uncomment ONE block (model + its dedicated port)
# =============================================================================

# 31B dense — flagship, best quality, ~20 GB, requires 24+ GB unified memory
MODEL="mlx-community/gemma-4-31b-it-4bit"
PORT="11440"

# 26B MoE (4B active) — efficient MoE variant, ~16 GB
# MODEL="mlx-community/gemma-4-26b-a4b-it-4bit"
# PORT="11441"

# E4B multimodal — compact, supports image/video/audio inputs, ~5 GB
# MODEL="mlx-community/gemma-4-e4b-it-8bit"
# PORT="11442"

# E2B multimodal — smallest, fast, supports image/video/audio inputs, ~3 GB
# MODEL="mlx-community/gemma-4-e2b-it-4bit"
# PORT="11443"

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

    info "Installing mlx-vlm..."
    activate_env
    pip install mlx-vlm

    info "Installation complete!"
    info ""
    info "Next steps:"
    info "  ./run-gemma-mlx.sh --serve"
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
    info "To remove them: rm -rf ~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit"
}

cmd_upgrade() {
    check_conda

    if ! env_exists; then
        error "Environment '${ENV_NAME}' does not exist. Run --install first."
    fi

    info "Upgrading packages in '${ENV_NAME}'..."
    activate_env
    pip install --upgrade mlx-vlm mlx

    info "Upgrade complete!"
}

cmd_serve() {
    check_conda

    if ! env_exists; then
        error "Environment '${ENV_NAME}' does not exist. Run --install first."
    fi

    info "Starting Gemma 4 server on port ${PORT}..."
    info "Model: ${MODEL}"
    info "Context: ${CONTEXT_LENGTH} tokens (128K)"
    info ""
    info "API endpoint: http://localhost:${PORT}/v1"
    info "Press Ctrl+C to stop"
    info ""

    activate_env
    exec python -m mlx_vlm.server \
        --model "$MODEL" \
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
