# AI Services

![Background](background.png)

A collection of useful AI services for AI sovereignty.

[![Video](./video.mp4)](https://github.com/user-attachments/assets/bc656f0d-6071-4ce4-b66b-e0c447435c66)

## Overview

This repository contains a set of containerized AI services that can be run locally to provide various AI capabilities without relying on external cloud providers. Each service is designed to be easy to deploy and use.

## Models

Local LLM inference with multiple backends (vLLM, llama.cpp, SGLang, MLX) and hardware targets (RTX PRO 6000, DGX Spark, AMD Vulkan, Apple Silicon).

| Model | Description | Location |
|-------|-------------|----------|
| **Qwen 3.5** | Flagship family, 0.8B to 122B variants | [models/qwen3.5](./models/qwen3.5) |
| **Qwen3-Coder-Next** | 80B MoE coding specialist | [models/qwen3-coder-next](./models/qwen3-coder-next) |
| **Qwopus** | Opus-reasoning distilled 27B | [models/qwopus](./models/qwopus) |
| **GLM-4.7-Flash** | 30B MoE, ~3.6B active params | [models/glm-4.7-flash](./models/glm-4.7-flash) |

Shared test and benchmark scripts live in [models/shared](./models/shared).

## Speech Services

| Service | Description | Location | Port |
|---------|-------------|----------|------|
| **Whisper** | Speech-to-text using OpenAI Whisper | [speech/whisper](./speech/whisper) | 8000 |
| **Faster Whisper** | Optimized Whisper variant | [speech/faster-whisper](./speech/faster-whisper) | — |
| **Orpheus TTS** | High-quality voice synthesis | [speech/orpheus](./speech/orpheus) | 5005 |

## Other Services

### Ollama

A server that runs large language models (LLMs) locally with GPU acceleration support.

- **Features**: Supports various open-source models, API access
- **Location**: [ollama](./ollama)
- **Port**: 11434

### Demo App (Voice Chat Assistant)

A real-time voice assistant integrating WebRTC, Whisper, Gemma 3, and Orpheus for end-to-end voice chat.

- **Location**: [demoapp](./demoapp)
- **Port**: 7860

## Getting Started

Each service has its own README.md with specific setup instructions and usage examples. Generally, you can start each service using:

```bash
cd service_directory
docker compose up -d
```

## Kudos and Credits

This project would not have been possible without the great works of many people who steadily contribute to the open source community!

- https://canopylabs.ai/
- https://github.com/Lex-au/Orpheus-FastAPI
- https://github.com/richardr1126/LlamaCpp-Orpheus-FastAPI
- https://github.com/freddyaboulton/fastrtc
- https://huggingface.co/
- https://ollama.com/
- https://www.gradio.app/
- https://www.langchain.com/

## System Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- Sufficient disk space for model storage

## License

See the [LICENSE](./LICENSE) file for details.
