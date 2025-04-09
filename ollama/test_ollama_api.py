#!/usr/bin/env python3
import os
import sys
import time
import ollama
from typing import List, Dict, Any

def main():
    # Set Ollama host if needed
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = ollama.Client(host=host)
    
    # Model to test
    model_name = "gemma3:27b"
    
    # Check if model exists
    models = client.list()
    model_exists = any(model["name"] == model_name for model in models.get("models", []))
    
    if not model_exists:
        pull_model(model_name)
    else:
        print(f"Model {model_name} already exists locally")
    
    # Test chat functionality
    test_chat(model_name)


def pull_model(model_name: str) -> None:
    """Pull the specified model from Ollama."""
    print(f"Pulling model {model_name}...")
    try:
        ollama.pull(model_name)
        print(f"Successfully pulled {model_name}")
    except ollama.ResponseError as e:
        print(f"Error pulling model: {e.error}")
        sys.exit(1)

def test_chat(model_name: str) -> None:
    """Test the chat functionality with the specified model."""
    print(f"\nTesting chat with {model_name}...")
    
    messages = [
        {
            'role': 'system',
            'content': 'You are a helpful AI assistant. Keep your answers brief and to the point.'
        },
        {
            'role': 'user',
            'content': 'What are the potential applications of large language models in healthcare?'
        }
    ]
    
    try:
        # Non-streaming response
        print("\n=== Non-streaming response ===")
        response = ollama.chat(model=model_name, messages=messages)
        print(f"Assistant: {response['message']['content']}")
        
        # Streaming response
        print("\n=== Streaming response ===")
        print("Assistant: ", end="", flush=True)
        stream = ollama.chat(
            model=model_name,
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            print(chunk['message']['content'], end="", flush=True)
        print("\n")
        
    except ollama.ResponseError as e:
        print(f"Error during chat: {e.error}")
        sys.exit(1)

if __name__ == "__main__":
    main()