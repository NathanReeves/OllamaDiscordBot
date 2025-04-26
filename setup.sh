#!/bin/bash

# Setup script for SignalOllamaBot

# Function to display usage information
show_help() {
  echo "SignalOllamaBot Setup Script"
  echo ""
  echo "Usage: ./setup.sh [command]"
  echo ""
  echo "Commands:"
  echo "  run      - Activate venv and run the bot"
  echo "  install  - Install dependencies in venv"
  echo "  help     - Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./setup.sh run     - Start the Discord bot"
  echo "  ./setup.sh install - Install dependencies"
}

# Check if venv exists, create if it doesn't
check_venv() {
  if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
  fi
}

# Install dependencies
install_dependencies() {
  check_venv
  echo "Installing dependencies..."
  source venv/bin/activate
  pip install -r requirements.txt
  echo "Dependencies installed successfully!"
}

# Run the bot
run_bot() {
  check_venv
  echo "Activating virtual environment and running bot..."
  source venv/bin/activate
  python bot.py
}

# Main script logic
case "$1" in
  "run")
    run_bot
    ;;
  "install")
    install_dependencies
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