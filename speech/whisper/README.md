# Whisper API

The Whisper API is a FastAPI server that provides speech-to-text transcription capabilities using OpenAI's Whisper models.

## Setup

To start the Whisper API service, run:

```bash
docker compose up -d
```

This will start the FastAPI server on port 8000 with GPU support (if available).

## Testing

To test the API, you can use the provided Python script:

```bash
python test_whisper_api.py testdata/recording.wav
```

### Additional Test Options

The test script supports several options:

```bash
python test_whisper_api.py testdata/recording.wav --model large
```

Available parameters:
- `--api-url`: Specify a custom API URL (default: http://localhost:8000)
- `--model`: Choose a specific model (options: tiny, base, small, medium, large, or 'all' to compare all models)

Example to test all available models:

```bash
python test_whisper_api.py testdata/recording.wav --model all
```

## API Endpoints

The service provides the following endpoints:

- `GET /`: API information
- `GET /health`: Health check
- `GET /models`: List available and loaded models
- `POST /transcribe/`: Transcribe audio file

### Transcription Endpoint

To transcribe an audio file, send a POST request with the file:

```bash
curl -F "file=@your_audio_file.wav" http://localhost:8000/transcribe/
```

You can specify a model with the `model_name` parameter:

```bash
curl -F "file=@your_audio_file.wav" "http://localhost:8000/transcribe/?model_name=large"
```

## Performance Metrics

The API provides metrics with each transcription, including:
- Audio length in seconds
- Transcription duration
- Transcription speed (real-time factor)

## Models

The service uses OpenAI's Whisper models and caches them in the models_cache directory.