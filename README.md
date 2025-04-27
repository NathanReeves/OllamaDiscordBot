# DiscordOllamaBot

A collaborative AI group chat bot for Discord, powered by Ollama.

Skills required:
- Be able to run terminal commands.
- Comfortable setting up a discord bot on your server. Great guide to dev portal [here](https://medium.com/technology-hits/how-to-create-a-discord-bot-514898ba0028).

## Install:
Install [Ollama](https://ollama.com/)
Install DiscordOllamaBot:
   ```sh
   # Install with default model (gemma3:12b)
   ./setup.sh install
   
   # Or install with a specific model
   ./setup.sh install llama3
   ```
## Setup .env
## Example:
    ```
    DISCORD_BOT_TOKEN=your_discord_bot_token_here
    OLLAMA_API_URL=http://localhost:11434
    ```
   
## Run:
   ```sh
   # Run with default or previously set model
   ./setup.sh run
   
   # Or run with a specific model
   ./setup.sh run llama3
   ```
   
   
## Discord Bot Linking:
Put your discord bot token in the .env before running.

   
## Discord Chat Commands:
!ai for any chat request.
!image to upload an image with your request (vision or multi-modal models only).
!history to see the current model context.
!model list available models or set model from Discord.

## Models:
Add new models to your library [here](https://ollama.com/search)
Change models at runtime from discord using the `!model` command.
View available models using `!model` without parameters.



Default model is `gemma3`.

Please feel free to use this in any way you like! 
Customize it, build on top of it for the coolest free and open source ai chat bot for your Discord.
