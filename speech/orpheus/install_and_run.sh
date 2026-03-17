#!/bin/bash

REPO_URL="git@github.com:richardr1126/LlamaCpp-Orpheus-FastAPI.git"
REPO_DIR="LlamaCpp-Orpheus-FastAPI"

# Check if the repository exists
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone "$REPO_URL"
else
    echo "Repository exists. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull
    cd ..
fi

# Copy the .env.example file to .env if it doesn't exist
if [ ! -d "$REPO_DIR/.env" ]; then
    echo "Creating .env file..."
    cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
else
    echo ".env file already exists."
fi

# Navigate into the repository directory and run docker-compose up
cd "$REPO_DIR"
echo "Starting Docker containers..."
docker compose up -d