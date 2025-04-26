# DiscordOllamaBot

A collaborative AI group chat bot for Discord, powered by Ollama.

## Install
1. Install [Ollama](https://ollama.com/)
2. Install DiscordOllamaBot:
   ```sh
   # Install with default model (gemma3:12b)
   ./setup.sh install
   
   # Or install with a specific model
   ./setup.sh install llama3
   ```
   
## Run
   ```sh
   # Run with default or previously set model
   ./setup.sh run
   
   # Or run with a specific model
   ./setup.sh run llama3
   ```

## .env Example
```
DISCORD_BOT_TOKEN=your_discord_bot_token_here
OLLAMA_API_URL=http://localhost:11434
```

## Models
The bot uses Ollama models. You can:
1. Specify a model during installation/run using the setup script
2. Change models at runtime using the `!model` command
3. View available models using `!model` without parameters

Default model is `gemma3:12b`