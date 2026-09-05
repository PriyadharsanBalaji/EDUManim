#!/bin/bash
# Keep Ollama alive script
# This script continuously checks if ollama is running. If not, it starts it in the background.

echo "Starting Ollama keep-alive service..."

while true; do
    if ! pgrep -x "ollama" > /dev/null; then
        echo "$(date): Ollama process not found. Starting ollama serve..."
        ollama serve &
    fi
    sleep 30
done
