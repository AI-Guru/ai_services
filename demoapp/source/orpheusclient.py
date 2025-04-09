import os
import requests
import tempfile
import logging
from typing import Optional, Union, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class OrpheusClient:
    """Client for the Orpheus Text-to-Speech API."""
    
    def __init__(self, api_url: str = None, voice: str = "alloy"):
        """
        Initialize the OrpheusClient.
        
        Args:
            api_url: Base URL for the Orpheus API. If None, will use environment variable ORPHEUS_API_URL.
            voice: Voice to use for synthesis. Default is "alloy".
        """
        self.api_url = api_url or os.environ.get("ORPHEUS_API_URL", "http://localhost:5005")
        self.voice = voice
        
        # Ensure API URL doesn't end with a trailing slash
        if self.api_url.endswith("/"):
            self.api_url = self.api_url[:-1]
            
        # Build endpoint URLs after removing trailing slash
        self.speech_endpoint = f"{self.api_url}/v1/audio/speech"
        self.voices_endpoint = f"{self.api_url}/v1/audio/voices"
            
        logger.debug(f"Initialized OrpheusClient with API URL: {self.api_url}")
    
    def get_available_voices(self) -> list:
        """
        Get a list of available voices from the API.
        
        Returns:
            List of available voice names.
        """
        try:
            response = requests.get(self.voices_endpoint)
            response.raise_for_status()
            return response.json().get("voices", [])
        except requests.RequestException as e:
            logger.error(f"Error fetching available voices: {e}")
            return []
    
    def synthesize(self, text: str, output_file: Optional[Union[str, Path]] = None, 
                  voice: Optional[str] = None, speed: float = 1.0) -> Optional[Path]:
        """
        Synthesize speech from text using the Orpheus API.
        
        Args:
            text: The text to synthesize.
            output_file: Path to save the audio file. If None, will create a temp file.
            voice: Voice to use. If None, will use the default voice from initialization.
            speed: Speech speed factor. Default is 1.0.
            
        Returns:
            Path to the saved audio file, or None if synthesis failed.
        """
        if not text:
            logger.warning("Empty text provided for synthesis")
            return None
            
        voice = voice or self.voice
        logger.debug(f"Synthesizing text with voice: {voice}")
        
        # Create temp file if no output path provided
        if output_file is None:
            temp_dir = tempfile.gettempdir()
            output_file = Path(temp_dir) / f"orpheus_speech_{voice}.wav"
        else:
            output_file = Path(output_file)
            
        # Prepare the request payload
        payload = {
            "input": text,
            "model": "orpheus",
            "voice": voice,
            "response_format": "wav",
            "speed": speed
        }
        
        try:
            # Make the API request
            response = requests.post(self.speech_endpoint, json=payload)
            response.raise_for_status()
            
            # Save the audio content to the output file
            with open(output_file, "wb") as f:
                f.write(response.content)
                
            logger.info(f"Speech synthesized and saved to: {output_file}")
            return output_file
            
        except requests.RequestException as e:
            logger.error(f"Error synthesizing speech: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"API response: {e.response.text}")
            return None

    def say(self, text: str, voice: Optional[str] = None) -> Optional[Path]:
        """
        Synthesize speech and return the path to the audio file.
        
        This is a convenience method that matches the interface expected by
        the talkapp module.
        
        Args:
            text: The text to synthesize.
            voice: Voice to use. If None, will use the default voice.
            
        Returns:
            Path to the saved audio file, or None if synthesis failed.
        """
        return self.synthesize(text, voice=voice)
