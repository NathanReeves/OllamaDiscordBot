# DiscordOllamaBot

A collaborative AI group chat bot for Discord, powered by Ollama.

## Skills required:
- Be able to run terminal commands.
- Comfortable setting up a discord bot on your server. Great guide to dev portal [here](https://medium.com/technology-hits/how-to-create-a-discord-bot-514898ba0028).

## Tools required:
- Ollama: [ollama](https://ollama.com/)
- Discord: [discord.com/download](https://discord.com/download)
- Discord Bot: [discord.com/developers](https://discord.com/developers/)

## Setup .env
    DISCORD_BOT_TOKEN=your_discord_bot_token_here
    OLLAMA_API_URL=http://localhost:11434

Install DiscordOllamaBot:
   ```sh
   # Install with default model (gemma3)
   ./setup.sh install
   
   # Or install with a specific model
   ./setup.sh install llama3
   ```
   
## Run:
   ```sh
   # Run with default or previously set model
   ./setup.sh run
   
   # Or run with a specific model
   ./setup.sh run llama3
   ```
   
   
## Discord Bot Linking:

Make sure your bot in Discord is working. Then put your discord bot token in the .env before running.
   
## Chat Commands:

`!ai` for any chat request.\
`!image` to upload an image with your request (vision or multi-modal models only).\
`!history` to see the current model context.\
`!model` list available models or set model from Discord.


## Models:

Add new models to your arsenal [here](https://ollama.com/search).\
Change models at runtime in Discord using the `!model` command.\
View available models using `!model` without parameters.

## Security note:
<b>Use at your own risk.</b> Hide your keys, hide your wife.\
Make the bot as private as you need to. [Discord Server Safety](https://discord.com/safety/360043653152-four-steps-to-a-super-safe-server)

## Have fun:
Please feel free to use this in any way you like. Build on top of it, make something cool.
