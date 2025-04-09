import requests
import json
from typing import Optional, Dict, Any, BinaryIO, Union, Tuple
import os
from pathlib import Path


class WhisperClient:
    """
    Client for interacting with the Whisper API for audio transcription.
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the WhisperClient.
        
        Args:
            base_url: The base URL of the Whisper API. If None, uses environment variable WHISPER_API_URL
                     or defaults to http://localhost:8000
            api_key: The API key for authentication. If None, uses environment variable WHISPER_API_KEY
        """
        self.base_url = base_url or os.environ.get("WHISPER_API_URL", "http://localhost:8000")
        self.api_key = api_key or os.environ.get("WHISPER_API_KEY")
        
        # Ensure base_url doesn't end with a slash
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests including authentication if available."""
        headers = {
            "Accept": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        return headers
            
    def transcribe(
        self, 
        file: Union[Tuple[str, bytes], BinaryIO, str, Path],
        model: str = "whisper-large-v3",
        response_format: str = "json",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        """
        Transcribe audio to text using the Whisper API.
        
        Args:
            file: Audio file to transcribe. Can be a tuple of (filename, bytes),
                 a file-like object, a string path, or a Path object
            model: The model to use for transcription
            response_format: The format for the response (json, text, verbose_json, etc.)
            language: Optional language code to specify the spoken language
            prompt: Optional prompt to guide the transcription
            temperature: Optional sampling temperature between 0 and 1
            
        Returns:
            The transcription response object
        """
        url = f"{self.base_url}/transcribe"
        
        files = {}
        data = {
            "model": model,
            "response_format": response_format,
        }
        
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        if temperature is not None:
            data["temperature"] = temperature
            
        # Handle different file input types
        if isinstance(file, tuple) and len(file) == 2:
            # (filename, bytes) format
            files["file"] = file[0]
        elif hasattr(file, 'read'):
            # File-like object
            files["file"] = file
        elif isinstance(file, (str, Path)):
            # File path
            path = Path(file)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file}")
            files["file"] = (path.name, path.read_bytes())
        else:
            raise ValueError("Invalid file format")
            
        headers = self._get_headers()
        # Don't include content-type in headers as requests will set it correctly for multipart/form-data
        if "Content-Type" in headers:
            del headers["Content-Type"]
            
        response = requests.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        
        result = response.json()
        
        # Create a response object similar to the Groq client for compatibility
        class TranscriptionResponse:
            def __init__(self, result):
                self.text = result.get("text", "")
                self._result = result
                
            def __getattr__(self, name):
                if name in self._result:
                    return self._result[name]
                raise AttributeError(f"'TranscriptionResponse' has no attribute '{name}'")
                
        return TranscriptionResponse(result)
