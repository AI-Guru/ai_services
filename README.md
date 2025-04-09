# AI Services

A collection of useful AI services for AI sovereignty.

## Overview

This repository contains a set of containerized AI services that can be run locally to provide various AI capabilities without relying on external cloud providers. Each service is designed to be easy to deploy and use.

## Services Included

### Ollama

A server that runs large language models (LLMs) locally with GPU acceleration support.

- **Features**: Supports various open-source models, API access
- **Location**: [/ollama](./ollama)
- **Port**: 11434

### Orpheus TTS

A text-to-speech (TTS) system powered by Llama-cpp backend with FastAPI.

- **Features**: High-quality voice synthesis, multiple voices, web interface
- **Location**: [/orpheus](./orpheus)
- **Port**: 5005

### Whisper API

A speech-to-text transcription service using OpenAI's Whisper models.

- **Features**: Multiple model sizes, real-time transcription metrics, audio file processing
- **Location**: [/whisper](./whisper)
- **Port**: 8000

## Getting Started

Each service has its own README.md with specific setup instructions and usage examples. Generally, you can start each service using:

```bash
cd service_directory
docker compose up -d
```

## System Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- Sufficient disk space for model storage

## License

See the [LICENSE](./LICENSE) file for details.
