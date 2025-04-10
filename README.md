# AI Services

![Background](background.png)

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

## Available Applications

### Demo App (Voice Chat Assistant)

Located in the `/demoapp` directory, this is a real-time voice assistant application that integrates WebRTC, speech-to-text, LLMs, and text-to-speech technologies.

The application provides a web interface for voice interactions with an AI assistant powered by Gemma 3. It demonstrates how multiple AI services can work together to create a conversational experience.

**Key Features:**
- Real-time speech recognition using Whisper
- Natural language processing with Gemma 3 LLM
- High-quality voice synthesis via Orpheus
- Browser-based interface using WebRTC

To get started with the Demo App, navigate to the `/demoapp` directory and follow the instructions in its README.md.

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
