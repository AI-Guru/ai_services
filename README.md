# AI Services

![Background](background.png)

A collection of useful AI services for AI sovereignty.

[![Video](./video.mp4)](https://github.com/user-attachments/assets/bc656f0d-6071-4ce4-b66b-e0c447435c66)

## Overview

This repository contains a set of containerized AI services that can be run locally to provide various AI capabilities without relying on external cloud providers. Each service is designed to be easy to deploy and use.

## Models

Local LLM inference with multiple backends (vLLM, llama.cpp, SGLang, MLX) and hardware targets (RTX PRO 6000, DGX Spark, AMD Vulkan, Apple Silicon). Each family lives under `models/<family>/` as a set of `docker-compose.<engine>-<variant>.yml` files serving an OpenAI-compatible API. See [models/README.md](./models/README.md) for the full variant/benchmark matrix and a "which variant should I use?" decision tree.

### Text generation & coding

| Model | Description | Location |
|-------|-------------|----------|
| **Qwen3.5** | Flagship family, 0.8B–122B dense/MoE variants | [models/qwen3.5](./models/qwen3.5) |
| **Qwen3.6** | Newer hybrid arch (Gated DeltaNet + Attention): 27B dense + 35B-A3B MoE | [models/qwen3.6](./models/qwen3.6) |
| **Qwen3-Coder-Next** | 80B MoE coding specialist (~3B active) | [models/qwen3-coder-next](./models/qwen3-coder-next) |
| **Qwopus** | Opus-reasoning distilled 27B dense | [models/qwopus](./models/qwopus) |
| **GLM-4.7-Flash** | 30B MoE, ~3.6B active params | [models/glm-4.7-flash](./models/glm-4.7-flash) |
| **Nemotron** | NVIDIA Cascade-2 / Nano family, hybrid Mamba-2 MoE (4B–120B) | [models/nemotron](./models/nemotron) |
| **Gemma 4** | Google, Apache 2.0, multimodal (text/image/audio), E2B–31B | [models/gemma4](./models/gemma4) |
| **Carnice-V2-27B** | Hermes-style agent SFT of Qwen3.6-27B | [models/carnice-v2](./models/carnice-v2) |
| **Mistral Medium 3.5** | Dense 128B, multimodal, 256K context | [models/mistral-medium-3.5](./models/mistral-medium-3.5) |

### Specialized

| Model | Description | Location |
|-------|-------------|----------|
| **Qwen3-Embedding & Reranker** | RAG building blocks — embeddings + reranking/scoring APIs | [models/qwen3-embedding](./models/qwen3-embedding) |
| **Qwen3-ASR** | Speech-to-text (52 languages) + forced aligner for timestamps | [models/qwen3-asr](./models/qwen3-asr) |
| **Qwen3Guard** | Generative safety classifier (Safe/Controversial/Unsafe, 119 languages) | [models/qwen3guard](./models/qwen3guard) |
| **DeepSeek-OCR** | Vision-LM, documents → markdown / HTML tables / LaTeX | [models/deepseek-ocr](./models/deepseek-ocr) |

Shared test and benchmark scripts live in [models/shared](./models/shared).

## Speech Services

| Service | Description | Location | Port |
|---------|-------------|----------|------|
| **Whisper** | Speech-to-text using OpenAI Whisper | [speech/whisper](./speech/whisper) | 8000 |
| **Faster Whisper** | Optimized Whisper variant | [speech/faster-whisper](./speech/faster-whisper) | — |
| **Orpheus TTS** | High-quality voice synthesis | [speech/orpheus](./speech/orpheus) | 5005 |

## Image Services

| Service | Description | Location | Port |
|---------|-------------|----------|------|
| **open-genmoji** | Custom emoji generation (Flux.1[dev] + LoRA, FP8 on Blackwell) | [open-genmoji](./open-genmoji) | 8888 |

## Monitoring

| Service | Description | Location | Port |
|---------|-------------|----------|------|
| **GPU Dashboard** | Grafana + Prometheus + nvidia_gpu_exporter for GPU metrics | [gpu-dashboard](./gpu-dashboard) | 3000 (Grafana), 9090 (Prometheus), 9835 (exporter) |
| **Netdata** | Real-time system & GPU monitoring with auto-detected NVIDIA metrics | [netdata](./netdata) | 19999 |

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
