import os
import boto3
from typing import Final
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()
bot_token = os.getenv('TELEGRAM_URL')

rekognition_client = boto3.client('rekognition')

# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey there, I am your friendly legal assistant.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("You have invoked the help command.")

# Helpers
def detect_text(image_bytes):
    print("===> Detecting text from image with AWS Rekognition")
    response = rekognition_client.detect_text(Image={'Bytes': image_bytes})
    detected_text = []
    for text_detection in response['TextDetections']:
        detected_text.append(text_detection['DetectedText'])
    return ' '.join(detected_text)

# Responses
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text: str = update.message.text
    print(f"===> User: {text}")

    response: str = "Thank you for telling me more about your situation."
    print(f"==> Bot: {response}")
    await update.message.reply_text(response)

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"===> User uploaded an image")
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


if __name__ == "__main__":
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
    print("===> Polling")
    app.run_polling(poll_interval=3)

