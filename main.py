import os
import boto3
from typing import Final
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate

load_dotenv()
bot_token = os.getenv('TELEGRAM_URL')


# Create a boto3 session with the new credentials
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

#bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)

modelID = "anthropic.claude-v2:1"


llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_tokens_to_sample": 2000,"temperature":0.9}
)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["language", "freeform_text"],
    template="You are a chatbot. You are in {language}.\n\n{freeform_text}"
)

# Initialize the LLMChain
bedrock_chain = LLMChain(llm=llm, prompt=prompt)

rekognition_client = boto3.client('rekognition')

# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey there, I am your friendly legal assistant.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("You have invoked the help command.")

# Helpers
def detect_text(image_bytes):
    print("==> Detecting text from image with AWS Rekognition")
    response = rekognition_client.detect_text(Image={'Bytes': image_bytes})
    detected_text = []
    for text_detection in response['TextDetections']:
        detected_text.append(text_detection['DetectedText'])
    return ' '.join(detected_text)

# Responses
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text: str = update.message.text
    print(f"==> User: {text}")

    language = "english"  # You can set this based on user preference
    freeform_text = text

    # Generate a response using Bedrock
    response = bedrock_chain({'language': language, 'freeform_text': freeform_text})
    generated_text = response['text']

    print(f"==> Bot: {generated_text}")
    await update.message.reply_text(generated_text)

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"==> User uploaded an image")
    file = await context.bot.getFile(update.message.photo[-1].file_id)
    image_bytes = await file.download_as_bytearray()
    
    detected_text = detect_text(image_bytes)
    if detected_text:
        await update.message.reply_text(f"Detected text:\n{detected_text}")
    else:
        await update.message.reply_text("No text detected.")

# Errors
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


if name == "main":
    print("=== igotscammed has been started ===")
    app = Application.builder().token(bot_token).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))

    # Responses
    app.add_handler(MessageHandler(filters.TEXT, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Errors
    app.add_error_handler(error)

    # Polls the bot
    print("==> Polling")
    app.run_polling(poll_interval=3)