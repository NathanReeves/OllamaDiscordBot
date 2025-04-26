#!/bin/bash

# Setup script for SignalOllamaBot

# Function to display usage information
show_help() {
  echo "SignalOllamaBot Setup Script"
  echo ""
  echo "Usage: ./setup.sh [command] [model]"
  echo ""
  echo "Commands:"
  echo "  run [model]      - Activate venv and run the bot with specified model"
  echo "  install [model]  - Install dependencies in venv and pull specified model"
  echo "  help            - Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./setup.sh run llama2    - Start the Discord bot with llama2 model"
  echo "  ./setup.sh install gemma  - Install dependencies and pull gemma model"
  echo "  ./setup.sh run           - Start with default model (gemma3:12b)"
}

# Check if venv exists, create if it doesn't
check_venv() {
  if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
  fi
}

# Pull Ollama model
pull_model() {
  local model=${1:-"gemma3:12b"}
  echo "Pulling Ollama model: $model..."
  ollama pull "$model"
}

# Install dependencies
install_dependencies() {
  local model=$1
  check_venv
  echo "Installing dependencies..."
  source venv/bin/activate
  pip install -r requirements.txt
  if [ ! -z "$model" ]; then
    pull_model "$model"
  fi
  echo "Dependencies installed successfully!"
}

# Run the bot
run_bot() {
  local model=$1
  check_venv
  if [ ! -z "$model" ]; then
    export OLLAMA_MODEL="$model"
    echo "Setting Ollama model to: $model"
  fi
  echo "Activating virtual environment and running bot..."
  source venv/bin/activate
  python bot.py
}

# Main script logic
case "$1" in
  "run")
    run_bot "$2"
    ;;
  "install")
    install_dependencies "$2"
    ;;
  "help"|"")
    show_help
    ;;
  *)
    echo "Unknown command: $1"
    show_help
    exit 1
    ;;
esac 