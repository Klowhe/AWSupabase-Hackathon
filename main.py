import os
import boto3
import json
import vecs
import backoff
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from openai import OpenAI, RateLimitError
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from supabase import create_client, Client

from aws_clients.rekognition_client import detect_text
from aws_clients.transcribe_client import transcribe_audio

# Load environment variables
load_dotenv()
bot_token = os.getenv("TELEGRAM_URL")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create a Supabase Client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase_client: Client = create_client(url, key)

# Create a vecs client
vx = vecs.create_client(os.getenv("DB_CONNECTION"))
sentences = vx.get_collection(name="sentences")

# Initialize Bedrock model ID and other settings
modelID = "anthropic.claude-v2:1"

# Create a boto3 session with the new credentials
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)

bedrock_client = session.client("bedrock-runtime")

# Initialize ChatBedrock
llm = ChatBedrock(
    model_id=modelID, client=bedrock_client, model_kwargs=dict(temperature=0.9)
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert guide specializing in Singapore's property laws , focusing on buying, selling, stamp duties, renovation disputes, neighbor conflicts, and home ownership issues. These are the relevant content that you can use to help answer the user's query: {context}. If necessary, ask the user clarifying questions to better understand his or her situation. Try to keep your replies succint as you are texting the user.",
        ),
        MessagesPlaceholder("history"),
        ("user", "{input}"),
    ]
)
bedrock_chain = prompt | llm
# Dictionary to hold users and their session ids
user_sessions = {}


# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_id = update.message.message_id
    session_id = message_id
    user_sessions[user_id] = session_id

    reply = "Hey there! I am SG Legal Guru (Home Edition). Please free to ask me any questions via text, voice message or even image uploads!"
    await update.message.reply_text(reply)

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "SG Legal Guru (Home Edition) is your go-to source for navigating Singapore's property legalities, including buying and selling processes, stamp duties, resolving renovation and neighbor disputes, and managing home ownership issues. It provides accurate insights and guidance to help you understand and address these specific legal topics with confidence, based on information obtained from wwww.singaporelegaladvice.com."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/about - Learn about SG Legal Guru (Home Edition)"
        "/start - Start a session to describe your situation\n"
        "/help - List of commands\n"
    )

# Helpers
@backoff.on_exception(backoff.expo, RateLimitError, max_time=300)
async def generate_embeddings_openai(text):
    response = bedrock_client.invoke_model(
        body=json.dumps({"inputText": text}),
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response["body"].read())
    query_embedding = response_body.get("embedding")
    return query_embedding

def store_message(user_id, message_id, role, content):
    (
        supabase_client.table("chat_messages")
        .insert(
            {
                "user_id": user_id,
                "message_id": message_id,
                "role": role,
                "content": content,
                "session_id": user_sessions[user_id],
            }
        )
        .execute()
    )

def retrieve_chat_history(user_id):
    response = (
        supabase_client.table("chat_messages")
        .select("role, content")
        .eq("user_id", user_id)
        .eq("session_id", user_sessions[user_id])
        .order("message_id", desc=False)
        .execute()
    )

    chat_history = response.data
    return chat_history

async def handle_message(user_id: str, text: str):
    embeddings = await generate_embeddings_openai(text)
    if embeddings:
        rag_results = sentences.query(data=embeddings, limit=3, include_value=True)
        print("rag results: ", rag_results)
        chat_history = retrieve_chat_history(user_id)
        response = bedrock_chain.invoke({"history": chat_history, "input": text, "context": rag_results})
        response_content = response.content
        return response_content
    else:
        return "Error generating embeddings. Please try again later."


# Responses
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_id = update.message.message_id
    text: str = update.message.text
    print(f"==> User: {text}")

    reply: str = await handle_message(user_id, text)
    await update.message.reply_text(reply)
    store_message(user_id, message_id, "user", text)
    store_message(user_id, message_id, "assistant", reply)


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_id = update.message.message_id
    print(f"==> User uploaded an image")

    # Download image file for Rekognition
    file = await context.bot.getFile(update.message.photo[-1].file_id)
    image_bytes = await file.download_as_bytearray()

    detected_text = detect_text(image_bytes)
    print(f"==> Detected text: {detected_text}")
    if detected_text == "":
        await update.message.reply_text(
            "No relevant text is detected from the image. Please try again."
        )
    else:
        reply: str = await handle_message(user_id, detected_text)
        await update.message.reply_text(reply)
        store_message(user_id, message_id, "user", detected_text)
        store_message(user_id, message_id, "assistant", reply)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_id = update.message.message_id
    print(f"==> User sent a voice message")

    # Download voice message for S3
    file_id = update.message.voice.file_id
    file = await context.bot.getFile(file_id)
    file_name = file.file_path.split(
        f"https://api.telegram.org/file/bot{bot_token}/voice/", 1
    )[1]
    await file.download_to_drive(custom_path=f"temp/{file_name}")

    transcript = transcribe_audio(file_name)
    print(f"==> Transcript: {transcript}")
    if transcript == "":
        await update.message.reply_text("Transcription failed. Please try again.")
    else:
        reply: str = await handle_message(user_id, transcript)
        await update.message.reply_text(reply)
        store_message(user_id, message_id, "user", transcript)
        store_message(user_id, message_id, "assistant", reply)


# Errors
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")


if __name__ == "__main__":
    print("=== telegram bot has been started ===")
    app = Application.builder().token(bot_token).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(CommandHandler("help", help_command))

    # Responses
    app.add_handler(MessageHandler(filters.TEXT, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Errors
    app.add_error_handler(error)

    # Polls the bot
    print("==> Polling")
    app.run_polling(poll_interval=3)
