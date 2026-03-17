import requests
import tempfile
from pathlib import Path
import os

def test_speech_synthesis():
    """
    Test the speech synthesis endpoint of the Orpheus TTS API.
    """
    # Configuration
    api_url = os.environ.get("ORPHEUS_API_URL", "http://localhost:5005")
    endpoint = f"{api_url}/v1/audio/speech"
    
    # Test text
    text = "Hello, this is a test of the Orpheus text-to-speech system."
    
    # Request payload
    payload = {
        "input": text,
        "model": "orpheus",
        "voice": "leah", # You can change to any available voice
        "response_format": "wav",
        "speed": 1.0
    }
    
    print(f"Sending request to {endpoint}")
    print(f"Payload: {payload}")
    
    try:
        # Make the API request
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        
        # Save the audio to a temporary file
        file_path = "orpheus_test_output.wav"
        with open(file_path, "wb") as f:
            f.write(response.content)
            
        print(f"Speech synthesized successfully!")
        print(f"Audio saved to: {file_path}")
        return file_path
        
    except requests.RequestException as e:
        print(f"Error synthesizing speech: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"API response: {e.response.text}")
        return None

if __name__ == "__main__":
    test_speech_synthesis()