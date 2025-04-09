# Ollama

Ollama is a server that runs on your machine and serves large language models (LLMs) locally. It provides a simple API for running inference with various open-source models.

## Setup

You can set it up like this:

```bash
docker compose up -d
```

This will start the Ollama service with GPU support (if available) on port 11434.

## Testing

You can test the API with the provided Python script:

```bash
python test_ollama_api.py
```

This script will automatically pull a model. This might take a while, depending on your internet connection and the model size.

## Configuration

The Docker Compose configuration includes:

- OLLAMA_HOST: Set to 0.0.0.0:11434 to allow external connections
- OLLAMA_CONTEXT_LENGTH: Set to 32000 tokens
- OLLAMA_KEEP_ALIVE: Models will stay loaded for 24 hours

## Available Models

After starting the server, you can pull models using the Ollama API or CLI. For example:

```bash
# Using curl
curl -X POST http://localhost:11434/api/pull -d '{"name": "llama3"}'

# Or with the ollama CLI (if installed locally)
ollama pull llama3
```

Visit the [Ollama documentation](https://github.com/ollama/ollama) for more information.