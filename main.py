import os
import time
import boto3
import json
from typing import Final
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()
bot_token = os.getenv('TELEGRAM_URL')

s3_client = boto3.client('s3')
rekognition_client = boto3.client('rekognition')
transcribe_client = boto3.client('transcribe')

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

    response: str = "Thank you for telling me more about your situation."
    print(f"==> Bot: {response}")
    await update.message.reply_text(response)

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"==> User uploaded an image")
    file = await context.bot.getFile(update.message.photo[-1].file_id)
    image_bytes = await file.download_as_bytearray()
    
    detected_text = detect_text(image_bytes)
    if detected_text:
        await update.message.reply_text(f"Detected text:\n{detected_text}")
    else:
        await update.message.reply_text("No text detected.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"==> User sent a voice message")
    # download voice msg to device
    file_id = update.message.voice.file_id
    file = await context.bot.getFile(file_id)
    await file.download_to_drive()
    file_path = file.file_path.split(f'https://api.telegram.org/file/bot{bot_token}/voice/', 1)[1]
    print("file path: ", file_path)

    # upload voice msg to s3
    s3_client.upload_file(file_path, 'voicemessagebucket', file_path)

    # start transcription job
    transcribe_client.start_transcription_job(
        TranscriptionJobName=f'{file_path}_transcription_job',
        LanguageCode='en-US',
        Media={
            'MediaFileUri': f's3://voicemessagebucket/{file_path}'
        },
        OutputBucketName='voicemessagetranscriptsbucket',
        OutputKey=f'{file_path}_transcript'
    )

    # query for transcription job status
    # possible transcription job statuses: QUEUED, IN_PROGRESS, FAILED, COMPLETED
    transcribe_status = transcribe_client.get_transcription_job(
        TranscriptionJobName=f'{file_path}_transcription_job'
    )['TranscriptionJob']['TranscriptionJobStatus']

    print("==> Transcribe Status: ", transcribe_status)
    
    while transcribe_status == 'IN_PROGRESS' or transcribe_status == 'QUEUED':
        transcribe_status = transcribe_client.get_transcription_job(
            TranscriptionJobName=f'{file_path}_transcription_job'
        )['TranscriptionJob']['TranscriptionJobStatus']
        print("==> Transcribe Status: ", transcribe_status)
        time.sleep(2)
    
    if transcribe_status == 'FAILED':
        await update.message.reply_text("Transcription of voice message failed.")
    else:
        print("==> Loading Transcript")
        # retrieving completed transcript from s3
        transcript_file = s3_client.get_object(
            Bucket='voicemessagetranscriptsbucket', 
            Key=f'{file_path}_transcript'
            )

        transcript_content = json.loads(transcript_file['Body'].read().decode('utf-8'))
        transcript = transcript_content['results']['transcripts'][0]['transcript']
        print("Transcript: ", transcript)
        await update.message.reply_text(f'Transcript: {transcript}')
    
    print("==> Cleaning up after transcription job")
    # removing voice message from device
    os.remove(file_path)
    # removing voice message from s3
    s3_client.delete_object(
        Bucket='voicemessagebucket', 
        Key=f'{file_path}'
    )
    # removing transcript from s3
    s3_client.delete_object(
        Bucket='voicemessagetranscriptsbucket', 
        Key=f'{file_path}_transcript'
    )
    print("==> Clean up completed")

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
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Errors
    app.add_error_handler(error)

    # Polls the bot
    print("==> Polling")
    app.run_polling(poll_interval=3)

