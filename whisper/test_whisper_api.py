#!/usr/bin/env python3
import requests
import argparse
import os
import sys
import time
from pathlib import Path
from prettytable import PrettyTable

def test_whisper_api(audio_file_path, api_url="http://localhost:8000", model=None):
    """
    Test the Whisper API by sending an audio file and receiving the transcription.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        api_url: Base URL of the Whisper API
        model: Optional model to use (tiny, base, small, medium, large, or "all" to test all models)
    
    Returns:
        The transcribed text if successful
    """
    # Check if the file exists
    if not os.path.isfile(audio_file_path):
        print(f"Error: File {audio_file_path} does not exist.")
        sys.exit(1)
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{api_url}/health")
        health_response.raise_for_status()
        print(f"Health check: {health_response.json()}")
        
        # Get root endpoint to check API info
        root_response = requests.get(f"{api_url}/")
        root_response.raise_for_status()
        print(f"API info: {root_response.json()}")
        
        # Check available models
        models_response = requests.get(f"{api_url}/models")
        models_response.raise_for_status()
        available_models = models_response.json()['available_models']
        loaded_models = models_response.json()['loaded_models']
        print(f"Available models: {available_models}")
        print(f"Currently loaded models: {loaded_models}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the API: {e}")
        print("Make sure the Whisper API server is running.")
        sys.exit(1)
    
    # If 'all' is specified, run transcription with all available models
    if model == "all":
        print(f"\n{'-' * 80}")
        print(f"Testing ALL available models with file: {audio_file_path}")
        print(f"{'-' * 80}")
        
        results_table = PrettyTable()
        results_table.field_names = ["Model", "Duration (s)", "Speed (x)", "Audio Length (s)"]
        
        comparison_results = []
        
        for m in available_models:
            print(f"\n{'-' * 30} Testing model: {m} {'-' * 30}")
            result = run_single_transcription(audio_file_path, api_url, m)
            
            # Store for comparison
            comparison_results.append({
                "model": m,
                "text": result["text"],
                "audio_length_seconds": result["audio_length_seconds"],
                "transcription_duration_seconds": result["transcription_duration_seconds"],
                "transcription_speed": result["transcription_speed"]
            })
            
            # Add to the table
            results_table.add_row([
                m,
                f"{result['transcription_duration_seconds']:.2f}",
                f"{result['transcription_speed']:.2f}",
                f"{result['audio_length_seconds']:.2f}"
            ])
        
        # Print performance comparison table
        print(f"\n{'-' * 80}")
        print("PERFORMANCE COMPARISON:")
        print(results_table)
        print(f"{'-' * 80}")
        
        # Print transcription results for each model
        print("\nTRANSCRIPTION RESULTS COMPARISON:")
        for result in comparison_results:
            print(f"\n{'=' * 20} Model: {result['model']} {'=' * 20}")
            print(result["text"])
            
        return comparison_results
    else:
        # Regular single model transcription
        result = run_single_transcription(audio_file_path, api_url, model)
        return result["text"]

def run_single_transcription(audio_file_path, api_url, model=None):
    """Run transcription with a single model and return the results"""
    # Prepare the file for upload
    files = {'file': (os.path.basename(audio_file_path), open(audio_file_path, 'rb'))}
    
    # Prepare params with optional model
    params = {}
    if model:
        params['model_name'] = model
        print(f"Using model: {model}")
    
    # Start timing
    start_time = time.time()
    
    # Send the transcription request
    try:
        print(f"Sending audio file {audio_file_path} to Whisper API...")
        response = requests.post(f"{api_url}/transcribe/", files=files, params=params)
        response.raise_for_status()
        
        result = response.json()
        elapsed_time = time.time() - start_time
        
        print(f"Transcription completed in {elapsed_time:.2f} seconds")
        print("\nTranscribed Text:")
        print("-" * 50)
        print(result["text"])
        print("-" * 50)
        
        # Print the new fields if they exist in the response
        if "audio_length_seconds" in result:
            print(f"Audio Length: {result['audio_length_seconds']} seconds")
        if "transcription_duration_seconds" in result:
            print(f"Transcription Duration: {result['transcription_duration_seconds']} seconds")
        if "transcription_speed" in result:
            print(f"Transcription Speed: {result['transcription_speed']}x (times faster than real-time)")
        if "model_used" in result:
            print(f"Model Used: {result['model_used']}")
        print("-" * 50)
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error during transcription: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"API response: {e.response.text}")
        sys.exit(1)
    finally:
        # Close the file
        files['file'][1].close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Whisper API with an audio file")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the Whisper API")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large", "all"], 
                        help="Model to use for transcription (tiny, base, small, medium, large, or 'all' to test all models)")
    
    args = parser.parse_args()
    test_whisper_api(args.audio_file, args.api_url, args.model)