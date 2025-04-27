import os
import discord
import requests
import aiohttp
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
import logging
import io
import re
from collections import defaultdict, deque
import time
import datetime
import asyncio
import json
import base64

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# Load environment variables from .env
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gemma3')
DEFAULT_BOT_NAME = os.getenv('DEFAULT_BOT_NAME', 'ollamabot')

logging.info(f"Initialized with model: {OLLAMA_MODEL}")

# Define system message for AI personality/behavior
SYSTEM_MESSAGE = """You are a helpful AI assistant in a Discord server. 
Your responses should be:
- Always concise and to the point
- Markdown-formatted for Discord

When writing code:
- Always wrap code in triple backticks with the language specified (e.g. ```python```)
- Include a filename comment at the start (e.g. # filename: example.py)
- Add clear comments and explanations

For complex technical responses:
- If you're explaining concepts, break them into digestible chunks
- If you're providing code, prefer outputting complete files

If you don't know something, say so directly."""

# Memory settings
MAX_HISTORY_LENGTH = 10  # Number of exchanges to remember
HISTORY_WINDOW = 15 * 60  # Consider messages from the last 15 minutes
HISTORY_EXPIRY = 60 * 60  # Remove messages older than 1 hour

# Conversation history storage
# Format: {channel_id: [(timestamp, role, content), ...]}
conversation_history = defaultdict(deque)

# Set up Discord bot with intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

def trim_history(channel_id):
    """Trim history to avoid exceeding token limits and remove old conversations."""
    history = conversation_history[channel_id]
    
    # Remove expired conversations
    current_time = time.time()
    while history and (current_time - history[0][0]) > HISTORY_EXPIRY:
        history.popleft()
        
    # Limit to most recent exchanges
    while len(history) > MAX_HISTORY_LENGTH * 2:  # *2 because each exchange has user and assistant messages
        history.popleft()

def format_history(channel_id):
    """Format the conversation history for the API call."""
    messages = []
    current_time = time.time()
    
    # Only include messages from the last HISTORY_WINDOW (15 minutes)
    for timestamp, role, content in conversation_history[channel_id]:
        if current_time - timestamp <= HISTORY_WINDOW:
            messages.append({"role": role, "content": content})
    
    return messages

# Helper function to update bot nickname
async def update_bot_nickname(guild, model_name, response_time=None):
    """Update bot nickname with model info and response time."""
    try:
        # Format the nickname
        if response_time:
            new_nickname = f"{DEFAULT_BOT_NAME} ({model_name}, {response_time:.1f}s)"
        else:
            new_nickname = f"{DEFAULT_BOT_NAME} ({model_name})"
            
        # Truncate if too long (Discord has a 32 character limit for nicknames)
        if len(new_nickname) > 32:
            new_nickname = new_nickname[:29] + "..."
            
        # Update the nickname in the guild
        await guild.me.edit(nick=new_nickname)
        logging.info(f"Updated bot nickname to: {new_nickname}")
    except Exception as e:
        logging.error(f"Failed to update bot nickname: {e}")

# Helper function to query Ollama
async def query_ollama(prompt, channel_id=None, guild=None):
    """Query Ollama API with memory of previous conversations using aiohttp."""
    model_name = OLLAMA_MODEL
    start_time = time.time()
    
    # Update bot nickname to show it's processing
    if guild:
        await update_bot_nickname(guild, model_name)
    
    if channel_id:
        # Trim and clean history
        trim_history(channel_id)
        
        # Add current prompt to history
        conversation_history[channel_id].append((time.time(), "user", prompt))
        
        # Format history for API
        messages = format_history(channel_id)
        
        url = f"{OLLAMA_API_URL}/api/chat"
        payload = {
            "model": model_name,
            "messages": messages,
            "system": SYSTEM_MESSAGE,
            "stream": True  # Enable streaming for chat API
        }
        
        logging.info(f"Sending request to Ollama with {len(messages)} messages in history")
    else:
        # Legacy fallback without history
        url = f"{OLLAMA_API_URL}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "system": SYSTEM_MESSAGE,
            "stream": True  # Enable streaming for generate API
        }
        logging.info(f"Sending request to Ollama without history: {prompt}")
    
    try:
        full_response = ""
        
        # Set a longer timeout (5 minutes) since we're streaming the response
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_msg = f"Error from Ollama API: HTTP {response.status}"
                    logging.error(error_msg)
                    return error_msg
                
                # Process streaming response
                async for line in response.content:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        if channel_id:  # Using chat API
                            if 'message' in data:
                                chunk = data['message'].get('content', '')
                                if chunk:
                                    full_response += chunk
                            # Done is when we get a message with 'done' set to true
                            if data.get('done', False):
                                break
                        else:  # Using generate API
                            chunk = data.get('response', '')
                            if chunk:
                                full_response += chunk
                            # Done is when we get a response with 'done' set to true
                            if data.get('done', False):
                                break
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse JSON from stream: {line}")
                        continue
                
                # Add the full response to history if using chat API
                if channel_id:
                    conversation_history[channel_id].append((time.time(), "assistant", full_response))
                    logging.info(f"Ollama streaming response completed and added to history")
                else:
                    logging.info(f"Ollama streaming response completed")
                
                # Calculate response time
                end_time = time.time()
                response_time = end_time - start_time
                
                # Update bot nickname with response time
                if guild:
                    await update_bot_nickname(guild, model_name, response_time)
                
                logging.info(f"Response time: {response_time:.2f} seconds")
                return full_response
                
    except asyncio.TimeoutError:
        error_msg = "Request to Ollama timed out after 5 minutes. The model might be generating a very long response."
        logging.error(error_msg)
        
        # Update nickname to show timeout
        if guild:
            await update_bot_nickname(guild, f"{model_name} (timeout)")
            
        return error_msg
    except Exception as e:
        error_msg = f"Error contacting Ollama: {e}"
        logging.error(error_msg)
        
        # Update nickname to show error
        if guild:
            await update_bot_nickname(guild, f"{model_name} (error)")
            
        return error_msg

# Helper function to query Ollama with an image
async def query_ollama_with_image(prompt, image_path, channel_id=None, guild=None):
    """Query Ollama API with an image input using aiohttp."""
    model_name = OLLAMA_MODEL
    start_time = time.time()
    
    logging.info(f"Processing image query - Model: {model_name}")
    
    # Update bot nickname to show it's processing
    if guild:
        await update_bot_nickname(guild, model_name)
    
    try:
        # Read and encode the image file
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            logging.info(f"Image encoded successfully ({len(image_data)} bytes)")
    except Exception as e:
        error_msg = f"Error reading image file: {str(e)}"
        logging.error(error_msg)
        return error_msg
    
    # Format the request for the chat API
    url = f"{OLLAMA_API_URL}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64_image]
            }
        ],
        "stream": True
    }
    
    logging.info("Sending request to Ollama API")
    
    try:
        full_response = ""
        chunk_count = 0
        
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_msg = f"Error from Ollama API: HTTP {response.status}"
                    error_body = await response.text()
                    logging.error(f"{error_msg}\nResponse body: {error_body}")
                    return f"{error_msg}\nResponse body: {error_body}"
                
                logging.info("Processing streaming response")
                async for line in response.content:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if 'message' in data:
                            chunk = data['message'].get('content', '')
                            if chunk:
                                full_response += chunk
                                chunk_count += 1
                        
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse JSON from stream: {line}")
                        continue
                
                end_time = time.time()
                response_time = end_time - start_time
                
                logging.info(f"Response completed in {response_time:.2f}s")
                
                if guild:
                    await update_bot_nickname(guild, model_name, response_time)
                
                if channel_id:
                    conversation_history[channel_id].append((time.time(), "user", prompt))
                    conversation_history[channel_id].append((time.time(), "assistant", full_response))
                
                return full_response
                
    except asyncio.TimeoutError:
        error_msg = "Request to Ollama timed out after 5 minutes"
        logging.error(f"{error_msg} - Model: {model_name}")
        if guild:
            await update_bot_nickname(guild, f"{model_name} (timeout)")
        return error_msg
    except Exception as e:
        error_msg = f"Error contacting Ollama: {str(e)}"
        logging.error(f"{error_msg} - Model: {model_name}")
        logging.exception("Full exception details:")
        if guild:
            await update_bot_nickname(guild, f"{model_name} (error)")
        return error_msg

@bot.event
async def on_ready():
    logging.info(f'Bot is ready and logged in as {bot.user}')
    
    # Set initial nickname
    for guild in bot.guilds:
        await update_bot_nickname(guild, OLLAMA_MODEL)
    
    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        logging.info(f"Synced {len(synced)} command(s)")
    except Exception as e:
        logging.error(f"Failed to sync commands: {e}")

def split_message(message, limit=1900):
    """Split a message into chunks that fit within Discord's message limit."""
    if len(message) <= limit:
        return [message]
    
    chunks = []
    current_chunk = ""
    
    # Split preferably at newlines, then at spaces, then just split at the limit
    for line in message.split('\n'):
        if len(current_chunk) + len(line) + 1 <= limit:
            current_chunk += line + '\n'
        else:
            # If the line itself is too long, split it at spaces
            if len(line) > limit:
                words = line.split(' ')
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= limit:
                        current_chunk += word + ' '
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = word + ' '
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_code_blocks(content):
    """Extract code blocks and their file names from the content."""
    # First find all code blocks with their language
    code_block_pattern = r"```([\w\+\-\.]*)\n(.*?)```"
    code_blocks = list(re.finditer(code_block_pattern, content, re.DOTALL))
    
    files = []
    detected_files = []
    
    # Start with removing all code blocks from the content
    content_without_code = re.sub(code_block_pattern, '', content, flags=re.DOTALL)
    
    for block in code_blocks:
        lang = block.group(1).strip() if block.group(1) else 'txt'
        code_content = block.group(2).strip()
        
        # Try to find filename in various comment formats
        filename_match = re.search(r'(?:^|\n)(?:#|//|/\*|<!--)\s*filename:\s*([^\n\r\*/>]+)', code_content)
        
        # Check for code file references in the message
        file_reference = None
        full_match = block.group(0)
        block_position = content.find(full_match)
        if block_position > 0:
            # Look for file references before the code block
            text_before = content[:block_position].strip()
            lines_before = text_before.split('\n')
            
            if lines_before:
                last_line = lines_before[-1].strip()
                # Check for common patterns like "filename.js:" or "In filename.js:"
                ref_patterns = [
                    r'([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+):$',  # filename.ext:
                    r'([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+)\s*\(?in a separate file\)?:$',  # filename.ext (in a separate file):
                    r'In\s+([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+):$',  # In filename.ext:
                    r'Create\s+([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+):$',  # Create filename.ext:
                ]
                
                for pattern in ref_patterns:
                    ref_match = re.search(pattern, last_line)
                    if ref_match:
                        file_reference = ref_match.group(1)
                        break
        
        if filename_match:
            filename = filename_match.group(1).strip()
            # Remove the filename line from the code
            code_content = re.sub(r'(?:^|\n)(?:#|//|/\*|<!--)\s*filename:\s*[^\n\r\*/>]+(?:\*/|-->)?\r?\n?', '\n', code_content, 1)
            detected_files.append(filename)
        elif file_reference:
            # Use the file reference from the message
            filename = file_reference
            detected_files.append(filename)
        else:
            # Try to generate a better filename based on content and language
            if lang.lower() in ['python', 'py']:
                # Try to find class or function name for Python
                class_match = re.search(r'class\s+([A-Za-z0-9_]+)', code_content)
                fn_match = re.search(r'def\s+([A-Za-z0-9_]+)', code_content)
                if class_match:
                    filename = f"{class_match.group(1).lower()}.py"
                elif fn_match:
                    filename = f"{fn_match.group(1).lower()}.py"
                else:
                    filename = f"script_{len(files) + 1}.py"
            elif lang.lower() in ['javascript', 'js']:
                # Try to find function or class name for JavaScript
                fn_match = re.search(r'function\s+([A-Za-z0-9_]+)', code_content)
                class_match = re.search(r'class\s+([A-Za-z0-9_]+)', code_content)
                if fn_match:
                    filename = f"{fn_match.group(1).lower()}.js"
                elif class_match:
                    filename = f"{class_match.group(1).lower()}.js"
                else:
                    filename = f"script_{len(files) + 1}.js"
            elif lang.lower() in ['html']:
                # Check for title in HTML
                title_match = re.search(r'<title>(.*?)</title>', code_content, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip().lower()
                    title = re.sub(r'[^\w\-]', '_', title)
                    filename = f"{title}.html"
                else:
                    filename = f"page_{len(files) + 1}.html"
            elif lang.lower() in ['css']:
                filename = f"styles_{len(files) + 1}.css"
            else:
                # Generic name based on language
                filename = f"code_snippet_{len(files) + 1}" + (f".{lang.lower()}" if lang else ".txt")
        
        # Clean up the code content
        code_content = code_content.strip()
        
        # Clean up the filename to be safe
        safe_filename = re.sub(r'[^\w\-\.]', '_', filename)
        if not safe_filename:
            safe_filename = f"code_snippet_{len(files) + 1}"
        
        files.append((safe_filename, code_content, lang))
    
    # If we found files, clean up references to them
    if files:
        # Remove common section headers
        headers_to_remove = [
            r'JavaScript \(in a separate file [^\)]+\):',
            r'JavaScript:',
            r'HTML:',
            r'CSS:',
            r'Python:',
            r'Java:',
        ]
        
        for header in headers_to_remove:
            content_without_code = re.sub(header, '', content_without_code)
        
        # Also try to remove references to the files we extracted
        for filename, _, _ in files:
            # Remove common file reference patterns
            patterns = [
                rf"Here['']?s the code for [`']?{re.escape(filename)}[`']?:?\s*",
                rf"I['']?ve created a file [`']?{re.escape(filename)}[`']?:?\s*",
                rf"Create a file called [`']?{re.escape(filename)}[`']?:?\s*",
                rf"In [`']?{re.escape(filename)}[`']?:?\s*",
                rf"[`']?{re.escape(filename)}[`']?:?\s*",
            ]
            for pattern in patterns:
                content_without_code = re.sub(pattern, '', content_without_code)
    
    # Clean up multiple newlines and whitespace
    content_without_code = re.sub(r'\n{3,}', '\n\n', content_without_code)
    content_without_code = content_without_code.strip()
    
    return content_without_code, files

@bot.tree.command(name="ai", description="Ask the AI a question")
@app_commands.describe(prompt="What would you like to ask?")
async def ai_command(interaction: discord.Interaction, prompt: str):
    logging.info(f"Received command from {interaction.user}: /ai {prompt}")
    
    # Use channel ID for context
    channel_id = str(interaction.channel_id)
    guild = interaction.guild
    
    # Acknowledge the interaction to prevent timeout
    await interaction.response.defer(thinking=True)
    
    try:
        response = await query_ollama(prompt, channel_id, guild)
        
        # Extract any code blocks with filenames
        message_content, code_files = extract_code_blocks(response)
        
        # Handle the main message content
        if message_content:
            if len(message_content) > 2000:
                chunks = split_message(message_content)
                await interaction.followup.send(chunks[0])
                for chunk in chunks[1:]:
                    await interaction.channel.send(chunk)
            else:
                await interaction.followup.send(message_content)
        elif not code_files:  # If no message content and no code files, send the original response
            if len(response) > 2000:
                chunks = split_message(response)
                await interaction.followup.send(chunks[0])
                for chunk in chunks[1:]:
                    await interaction.channel.send(chunk)
            else:
                await interaction.followup.send(response)
        else:
            # If we only have files but no text content, add a simple message
            await interaction.followup.send("Here are the files you requested:")
        
        # Handle any code files
        for filename, code, lang in code_files:
            # Ensure filename has an appropriate extension if it doesn't have one
            if '.' not in filename:
                ext = f".{lang.lower()}" if lang.lower() not in ['txt', ''] else '.txt'
                filename = filename + ext
            
            # Create and send the file
            file = io.StringIO(code)
            discord_file = discord.File(fp=file, filename=filename)
            await interaction.channel.send(f"📄 `{filename}`", file=discord_file)
            file.close()
            
        logging.info(f"Replied to {interaction.user} with response and {len(code_files)} code files")
    except Exception as e:
        logging.error(f"Error in ai_command: {e}")
        await interaction.followup.send(f"An error occurred: {e}")

# Model command
@bot.tree.command(name="model", description="Set or show the current Ollama model")
@app_commands.describe(model_name="Name of the model to use (leave empty to show current model)")
async def model_command(interaction: discord.Interaction, model_name: str = None):
    """Set or show the current Ollama model"""
    global OLLAMA_MODEL
    
    await interaction.response.defer(thinking=True)
    
    if model_name:
        # Check if model exists
        try:
            url = f"{OLLAMA_API_URL}/api/tags"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        available_models = [model['name'] for model in data.get('models', [])]
                        
                        # Create case-insensitive lookup dictionary
                        model_lookup = {m.lower(): m for m in available_models}
                        
                        # First try exact match (case-insensitive)
                        if model_name.lower() in model_lookup:
                            old_model = OLLAMA_MODEL
                            actual_model = model_lookup[model_name.lower()]
                            OLLAMA_MODEL = actual_model
                            await update_bot_nickname(interaction.guild, model_name)
                            await interaction.followup.send(f"Model changed from `{old_model}` to `{actual_model}`")
                            logging.info(f"Model changed to {model_name}")
                            return
                        
                        # If no exact match, try with ":latest" suffix (case-insensitive)
                        model_with_latest = f"{model_name}:latest"
                        if model_with_latest.lower() in model_lookup:
                            old_model = OLLAMA_MODEL
                            actual_model = model_lookup[model_with_latest.lower()]
                            OLLAMA_MODEL = actual_model
                            await update_bot_nickname(interaction.guild, model_name)
                            await interaction.followup.send(f"Model changed from `{old_model}` to `{actual_model}`")
                            logging.info(f"Model changed to {model_name}")
                            return
                            
                        # If still no match, check if any version exists (case-insensitive)
                        matching_models = [m for m in available_models if m.lower().startswith(f"{model_name.lower()}:")]
                        if matching_models:
                            # Show available versions
                            models_list = "\n".join([f"- `{m}`" for m in matching_models])
                            await interaction.followup.send(f"Multiple versions of `{model_name}` found. Please specify which version you want:\n{models_list}")
                        else:
                            # Show available models
                            models_list = "\n".join([f"- `{m}`" for m in available_models])
                            await interaction.followup.send(f"Model `{model_name}` not found. Available models:\n{models_list}")
                    else:
                        # If can't check models, try to set it anyway
                        OLLAMA_MODEL = model_name
                        await update_bot_nickname(interaction.guild, model_name)
                        await interaction.followup.send(f"Model set to `{model_name}` (availability not verified)")
        except Exception as e:
            logging.error(f"Error checking models: {e}")
            # If error, try to set the model anyway
            OLLAMA_MODEL = model_name
            await update_bot_nickname(interaction.guild, model_name)
            await interaction.followup.send(f"Model set to `{model_name}` (availability not verified)")
    else:
        # Show current model
        await interaction.followup.send(f"Current model: `{OLLAMA_MODEL}`")
        
        # Try to list available models
        try:
            url = f"{OLLAMA_API_URL}/api/tags"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        if models:
                            # Show all models without truncation
                            models_list = "\n".join([f"- `{model['name']}` ({model['size']})" for model in models])
                            await interaction.channel.send(f"Available models:\n{models_list}")
        except Exception as e:
            logging.error(f"Error listing models: {e}")

# Reset command
@bot.tree.command(name="reset", description="Reset the conversation history")
async def reset_command(interaction: discord.Interaction):
    channel_id = str(interaction.channel_id)
    
    await interaction.response.defer(thinking=True)
    
    if channel_id in conversation_history:
        conversation_history[channel_id].clear()
        await interaction.followup.send("Conversation history has been reset!")
        logging.info(f"Conversation history reset for channel {channel_id}")
    else:
        await interaction.followup.send("No conversation history to reset.")

# History command
@bot.tree.command(name="history", description="View recent conversation history")
async def history_command(interaction: discord.Interaction):
    channel_id = str(interaction.channel_id)
    
    await interaction.response.defer(thinking=True)
    
    if channel_id not in conversation_history or not conversation_history[channel_id]:
        await interaction.followup.send("No conversation history in this channel.")
        return
    
    # Create embed for history
    embed = discord.Embed(
        title="Recent Conversation History",
        description="Messages considered for context in the last 15 minutes",
        color=discord.Color.blue()
    )
    
    # Format timestamps and add fields to embed
    current_time = time.time()
    count = 0
    
    # Show the most recent messages first (up to 10)
    history_list = list(conversation_history[channel_id])
    history_list.reverse()  # Most recent first
    
    for timestamp, role, content in history_list:
        # Skip messages older than 15 minutes
        if current_time - timestamp > HISTORY_WINDOW:
            continue
            
        # Format timestamp
        dt = datetime.datetime.fromtimestamp(timestamp)
        time_str = dt.strftime("%H:%M:%S")
        
        # Truncate content if too long
        if len(content) > 250:
            content = content[:247] + "..."
        
        # Add field to embed
        role_name = "🧑 User" if role == "user" else "🤖 Bot"
        embed.add_field(
            name=f"{role_name} ({time_str})",
            value=content,
            inline=False
        )
        
        count += 1
        if count >= 10:  # Limit to 10 entries
            break
    
    if count == 0:
        embed.add_field(
            name="No Recent Messages",
            value="There are no messages in the 15-minute context window.",
            inline=False
        )
    else:
        # Add footer with context info
        embed.set_footer(text=f"Showing {count} messages in the context window | Bot will use these for context")
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="image", description="Analyze the most recent image")
@app_commands.describe(prompt="What would you like to ask about the image?")
async def image_command(interaction: discord.Interaction, prompt: str = None):
    """Ask the ai about an image. Uses the most recent image in the channel."""
    logging.info(f"Image command received from {interaction.user}")
    
    await interaction.response.defer(thinking=True)
    
    # Search for the most recent image in the channel
    channel = interaction.channel
    image_found = False
    last_image_url = None
    last_image_filename = "unknown.jpg"
    
    try:
        # Look through recent messages to find an image
        async for message in channel.history(limit=20):  # Check the last 20 messages
            if message.attachments:
                for attachment in message.attachments:
                    # Check if attachment is an image by content type or extension
                    if attachment.content_type and attachment.content_type.startswith('image/'):
                        last_image_url = attachment.url
                        last_image_filename = attachment.filename
                        image_found = True
                        logging.info(f"Found image in message from {message.author}: {last_image_filename}")
                        break
                if image_found:
                    break
        
        if not image_found:
            await interaction.followup.send("No recent images found in this channel. Please upload an image first, then use the /image command.")
            return
        
        # Create temp_images directory if it doesn't exist
        os.makedirs('temp_images', exist_ok=True)
        
        # Save the image with a unique filename to prevent race conditions
        timestamp = int(time.time())
        image_path = os.path.join('temp_images', f'image_{timestamp}_{interaction.user.id}.jpg')
        
        # Download the image
        async with aiohttp.ClientSession() as session:
            async with session.get(last_image_url) as resp:
                if resp.status == 200:
                    image_data = await resp.read()
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    logging.info(f"Downloaded image to: {image_path}")
                else:
                    await interaction.followup.send(f"Error downloading image: HTTP status {resp.status}")
                    return
        
        # Get the prompt
        if not prompt:
            prompt = "Please describe this image in detail."
        
        try:
            response = await query_ollama_with_image(prompt, image_path, interaction.channel_id, interaction.guild)
            
            # Split long responses into chunks
            if len(response) > 2000:
                chunks = split_message(response)
                await interaction.followup.send(chunks[0])
                for chunk in chunks[1:]:
                    await interaction.channel.send(chunk)
                    await asyncio.sleep(0.5)  # Add small delay between chunks
            else:
                await interaction.followup.send(response)
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            logging.error(error_msg)
            logging.exception("Full exception details:")
            await interaction.followup.send(error_msg)
        
        finally:
            # Try to clean up the image file
            try:
                os.remove(image_path)
                logging.info(f"Removed temporary image: {image_path}")
            except Exception as e:
                logging.error(f"Failed to remove temporary image: {e}")
                
    except Exception as e:
        error_msg = f"Error finding or processing image: {str(e)}"
        logging.error(error_msg)
        logging.exception("Full exception details:")
        await interaction.followup.send(error_msg)

if __name__ == '__main__':
    bot.run(DISCORD_BOT_TOKEN) 
