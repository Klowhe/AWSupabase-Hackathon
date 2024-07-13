import os
import boto3
import backoff
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI, RateLimitError
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate

from aws_clients.rekognition_client import detect_text
from aws_clients.transcribe_client import transcribe_audio

# Load environment variables
load_dotenv()
bot_token = os.getenv('TELEGRAM_URL')

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Bedrock model ID and other settings
modelID = "anthropic.claude-v2:1"

# Create a boto3 session with the new credentials
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

bedrock_client = session.client("bedrock-runtime")

# Initialize ChatBedrock
llm = ChatBedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs=dict(temperature=0.9)
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a chatbot. You are in {language}."),
        ("human", "{input}")
    ]
)
bedrock_chain = prompt | llm

# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey there, I am your friendly legal assistant.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("You have invoked the help command.")

# Helpers
@backoff.on_exception(backoff.expo, RateLimitError, max_time=300)
async def generate_embeddings_openai(text):
    text = text.replace("\n", " ")
    response = openai_client.embeddings.create(input=[text], model="text-embedding-ada-002")
    if response and response.data and len(response.data) > 0:
        return response.data[0].embedding
    return None

def handle_message(text: str):
    # You can set this based on user preference
    language = "english"  
    response = bedrock_chain.invoke({'language': language, 'input': text })
    response_content = response.content
    return response_content

# Responses
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text: str = update.message.text
    print(f"==> User: {text}")

    # Step 1: Generate embeddings for user input
    embeddings = await generate_embeddings_openai(text)

    if embeddings:
        print(f"==> Embeddings: {embeddings}")

        # Step 2: Use embeddings for further processing (querying Bedrock)
        reply: str = handle_message(text)
        await update.message.reply_text(reply)
    else:
        await update.message.reply_text("Error generating embeddings. Please try again later.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"==> User uploaded an image")

    # Download image file for Rekognition
    file = await context.bot.getFile(update.message.photo[-1].file_id)
    image_bytes = await file.download_as_bytearray()

    detected_text = detect_text(image_bytes)
    print(f"==> Detected text: {detected_text}")
    if detected_text == '':
        await update.message.reply_text("No relevant text is detected from the image. Please try again.")
    else:
        reply: str = handle_message(detected_text)
        await update.message.reply_text(reply)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"==> User sent a voice message")
    
    # Download voice message for S3
    file_id = update.message.voice.file_id
    file = await context.bot.getFile(file_id)
    file_name = file.file_path.split(f'https://api.telegram.org/file/bot{bot_token}/voice/', 1)[1]
    await file.download_to_drive(custom_path=f"temp/{file_name}")

    transcript = transcribe_audio(file_name)
    print(f"==> Transcript: {transcript}")
    if transcript == '':
        await update.message.reply_text("Transcription failed. Please try again.")
    else:
        reply: str = handle_message(transcript)
        await update.message.reply_text(reply)

# Errors
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


if __name__ == "__main__":
    print("=== telegram bot has been started ===")
    app = Application.builder().token(bot_token).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))

    # Responses
    app.add_handler(MessageHandler(filters.TEXT, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Errors
    app.add_error_handler(error)

    # Polls the bot
    print("==> Polling")
    app.run_polling(poll_interval=3)
