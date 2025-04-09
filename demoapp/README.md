# Voice Chat Demo App

A real-time voice assistant using WebRTC, speech-to-text, LLM, and text-to-speech technologies.

## About Demo App

This application (demoapp) provides a simple web interface for real-time voice interactions with an AI assistant. It demonstrates the integration of various AI services into a cohesive conversational experience.

## Requirements

- Python 3.10+
- Running services:
  - Whisper API (port 8000)
  - Ollama with gemma3:27b model
  - Orpheus API (port 5005)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python demoapp.py
```

Open http://127.0.0.1:7860 in your browser and start talking.

## How It Works

1. WebRTC captures your voice from the browser
2. Whisper API transcribes speech to text
3. Gemma 3 LLM generates responses
4. Orpheus synthesizes speech output