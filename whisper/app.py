import io
import os
import torch
import time
import tempfile
import subprocess
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import uvicorn
import logging
from typing import Dict, Optional, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper Speech Recognition API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model manager class to handle multiple models
class WhisperModelManager:
    def __init__(self):
        self.models: Dict[str, dict] = {}
        self.default_model_id = os.environ.get("WHISPER_MODEL_ID", "openai/whisper-large-v3")
        self.available_models = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v3",
        }
        self.cache_dir = os.environ.get("HF_HOME", None)
    
    def get_model_pipeline(self, model_name: str = None) -> dict:
        """
        Get or load a model pipeline by name
        
        Args:
            model_name: The name of the model to load (tiny, base, small, medium, large)
                        If None, uses the default model
                        
        Returns:
            The loaded pipeline
        """
        # Determine which model ID to use
        if model_name and model_name in self.available_models:
            model_id = self.available_models[model_name]
        else:
            # If model name not specified or invalid, use default
            model_id = self.default_model_id
            model_name = next((k for k, v in self.available_models.items() if v == model_id), "large")
        
        # Check if model is already loaded
        if model_name in self.models:
            logger.info(f"Using already loaded model: {model_name}")
            return self.models[model_name]
        
        # Load the model
        logger.info(f"Loading Whisper model: {model_name} ({model_id})")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Using device: {device}, dtype: {torch_dtype}")
        if self.cache_dir:
            logger.info(f"Using cache directory: {self.cache_dir}")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            cache_dir=self.cache_dir
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        # Store the loaded model
        self.models[model_name] = {
            "pipeline": pipe,
            "model_id": model_id,
            "last_used": time.time()
        }
        
        logger.info(f"Model {model_name} loaded successfully")
        return self.models[model_name]
    
    def list_available_models(self) -> List[str]:
        """Return a list of available model names"""
        return list(self.available_models.keys())
    
    def list_loaded_models(self) -> List[str]:
        """Return a list of currently loaded model names"""
        return list(self.models.keys())

# Create the model manager
model_manager = WhisperModelManager()

def convert_audio_to_wav(audio_bytes: bytes, original_filename: str) -> tuple:
    """
    Convert audio file to WAV format using ffmpeg if needed.

    Args:
        audio_bytes: Raw audio file bytes
        original_filename: Original filename (used to determine format)

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    # Extract file extension for logging and temp file creation
    file_ext = os.path.splitext(original_filename)[1].lower() if original_filename else ''

    try:
        # First, try to read directly with soundfile
        audio_buffer = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(audio_buffer)
        logger.info(f"Successfully read audio directly with soundfile (file: {original_filename})")
        return data, samplerate
    except Exception as e:
        logger.info(f"Direct read failed for {original_filename}: {e}. Attempting conversion with ffmpeg...")

        # If direct read fails, use ffmpeg to convert
        try:
            # Use the original file extension if available, otherwise default to .audio
            input_suffix = file_ext if file_ext else '.audio'

            with tempfile.NamedTemporaryFile(suffix=input_suffix, delete=False) as input_file:
                input_path = input_file.name
                input_file.write(audio_bytes)

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_file:
                output_path = output_file.name

            # Convert using ffmpeg
            # Note: ffmpeg will auto-detect format from extension and file content
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ar', '16000',  # Resample to 16kHz (optimal for Whisper)
                '-ac', '1',       # Convert to mono
                '-f', 'wav',      # Output format
                '-y',             # Overwrite output file
                output_path
            ]

            logger.info(f"Converting {original_filename} ({file_ext or 'unknown format'}) to WAV using ffmpeg...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg error for {original_filename}: {result.stderr}")
                raise Exception(f"FFmpeg conversion failed for {original_filename}: {result.stderr}")

            # Read the converted WAV file
            data, samplerate = sf.read(output_path)
            logger.info(f"Successfully converted {original_filename} ({file_ext}) with ffmpeg to 16kHz mono WAV")

            # Clean up temporary files
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp files: {cleanup_error}")

            return data, samplerate

        except subprocess.TimeoutExpired:
            raise Exception(f"Audio conversion timed out for {original_filename}")
        except Exception as conversion_error:
            logger.error(f"Audio conversion failed for {original_filename}: {conversion_error}")
            raise Exception(f"Failed to process audio file {original_filename}: {conversion_error}")

# Initialize the default model
@app.on_event("startup")
async def startup_event():
    # Pre-load the default model on startup
    model_manager.get_model_pipeline()
    logger.info("Whisper API started successfully")

@app.get("/")
async def root():
    cache_dir = os.environ.get("HF_HOME", "default cache")
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    return {
        "message": "Whisper Speech Recognition API",
        "default_model": model_manager.default_model_id,
        "available_models": model_manager.list_available_models(),
        "loaded_models": model_manager.list_loaded_models(),
        "device": device,
        "cache_location": cache_dir
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/models")
async def list_models():
    """List all available and loaded models"""
    return {
        "available_models": model_manager.list_available_models(),
        "loaded_models": model_manager.list_loaded_models(),
    }

@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_name: Optional[str] = Query(None, description="Model to use for transcription (tiny, base, small, medium, large)")
):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Get the appropriate model pipeline
        model_data = model_manager.get_model_pipeline(model_name)
        pipe = model_data["pipeline"]
        used_model = model_data["model_id"]

        # Update last used timestamp
        model_data["last_used"] = time.time()

        # Read the uploaded file
        contents = await file.read()

        # Convert audio to WAV format if needed (handles .oga, .mp3, .m4a, etc.)
        data, samplerate = convert_audio_to_wav(contents, file.filename or "audio.wav")

        # Calculate audio length in seconds
        audio_length_seconds = len(data) / samplerate

        # Measure transcription time
        start_time = time.time()

        # Process with Whisper
        result = pipe({"raw": data, "sampling_rate": samplerate})

        # Calculate transcription duration
        transcription_duration = time.time() - start_time

        # Calculate transcription speed (ratio of audio length to processing time)
        transcription_speed = audio_length_seconds / transcription_duration if transcription_duration > 0 else 0

        return {
            "text": result["text"],
            "audio_length_seconds": round(audio_length_seconds, 2),
            "transcription_duration_seconds": round(transcription_duration, 2),
            "transcription_speed": round(transcription_speed, 2),  # times faster than real-time
            "model_used": used_model,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)