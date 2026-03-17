# Orpheus TTS with FastAPI

Orpheus is a text-to-speech (TTS) system that uses the Llama-cpp backend with FastAPI to provide high-quality voice synthesis.

## Setup

The easiest way to get started is to run the provided installation script:

```bash
sh install_and_run.sh
```

This script will:
1. Clone the LlamaCpp-Orpheus-FastAPI repository (if not already present)
2. Create an `.env` file from `.env.example` (if needed)
3. Start the Docker containers with docker compose

## Configuration

You can customize the service by editing the `.env` file in the LlamaCpp-Orpheus-FastAPI directory. After making changes, restart the containers:

```bash
cd LlamaCpp-Orpheus-FastAPI
docker compose up -d
```

## Testing the API

You can test the API using the provided Python script:

```bash
python test_orpheus_api.py
```

This will generate a sample audio file named `orpheus_test_output.wav` with synthesized speech.

## API Usage

The API provides a speech synthesis endpoint at `/v1/audio/speech` that accepts the following parameters:
- `input`: The text to convert to speech
- `model`: Model to use (default is "orpheus")
- `voice`: Voice to use (available options include "leah")
- `response_format`: Audio format (default is "wav")
- `speed`: Speech speed multiplier (default is 1.0)

## Web Interface

A web interface is available at http://localhost:5005 for testing the TTS capabilities directly in your browser.