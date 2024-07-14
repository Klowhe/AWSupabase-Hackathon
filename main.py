import os
import boto3
import backoff
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI, RateLimitError
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from supabase import create_client, Client

from aws_clients.rekognition_client import detect_text
from aws_clients.transcribe_client import transcribe_audio
import json
import vecs

# Load environment variables
load_dotenv()
bot_token = os.getenv('TELEGRAM_URL')

DB_CONNECTION = os.getenv("DATABASE_URL")
# create vector store client
vx = vecs.Client(DB_CONNECTION)
# to match the default dimension of the Titan Embeddings G1 - Text model
vectordb = vx.get_or_create_collection(name="vectordb", dimension=1536)

# Create a Supabase Client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase_client: Client = create_client(url, key)

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
        ("system", "You are an expert guide specializing in Singapore's property laws , focusing on buying, selling, stamp duties, renovation disputes, neighbor conflicts, and home ownership issues."),
        MessagesPlaceholder("history"),
        ("user", "{input}")
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

    reply = "Hey there, I am your friendly legal assistant."
    await update.message.reply_text(reply)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("You have invoked the help command.")

def get_text_embedding(text: str):
    try:
        response = bedrock_client.invoke_model(
            body=json.dumps({"inputText": text}),
            modelId="amazon.titan-embed-text-v1",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response["body"].read())
        embeddings = response_body.get("embedding")
        return embeddings
    except Exception as e:
        print(f"Error generating text embeddings: {e}")
        return None

def get_image_embedding(image_bytes: bytes):
    try:
        response = bedrock_client.invoke_model(
            body=json.dumps({"inputImage": image_bytes.decode('latin-1')}),  # Convert bytes to base64 string
            modelId="amazon.titan-embed-image-v1",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response["body"].read())
        embeddings = response_body.get("embedding")
        return embeddings
    except Exception as e:
        print(f"Error generating image embeddings: {e}")
        return None

async def query_supabase_for_relevant_content(embedding):
    query = """
    SELECT * FROM vectordb
    ORDER BY similarity(embedding, %s) DESC
    LIMIT 5
    """
    results = await vx.query(query, (embedding,))
    return results

def get_claude_response(question: str, context: str):
    prompt = f"Context: {context}\nQuestion: {question}"
    try:
        response = bedrock_client.invoke_model(
            body=json.dumps({"content": prompt}),
            modelId="anthropic.claude-v2:1",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response["body"].read())
        answer = response_body.get("content")
        return answer
    except Exception as e:
        print(f"Error generating Claude response: {e}")
        return "An error occurred while generating the response."

def store_message(user_id, message_id, role, content):
    (supabase_client.table("chat_messages")
     .insert({
         "user_id": user_id,
         "message_id": message_id,
         "role": role,
         "content": content,
         "session_id": user_sessions[user_id]
     }).execute())
    
def retrieve_chat_history(user_id):
    response = (supabase_client.table("chat_messages") 
                       .select("role, content") 
                       .eq("user_id", user_id) 
                       .eq("session_id", user_sessions[user_id]) 
                       .order("message_id", desc=False)
                       .execute())
    
    chat_history = response.data
    return chat_history

def handle_message(user_id: str, text: str):
    chat_history = retrieve_chat_history(user_id)
    response = bedrock_chain.invoke(
        {
            "history": chat_history,
            "input": text
        }  
    )
    response_content = response.content
    return response_content

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_id = update.message.message_id
    text: str = update.message.text
    print(f"==> User: {text}")
    print(f"==>hey")

    embeddings = get_text_embedding(text)
    if embeddings:
        print(f"==> Embeddings: {embeddings}")
        relevant_content = await query_supabase_for_relevant_content(embeddings)
        context_text = " ".join(item['text'] for item in relevant_content)  # Adjust as needed
        reply = get_claude_response(text, context_text)
        await update.message.reply_text(reply)
        store_message(user_id, message_id, "user", text)
        store_message(user_id, message_id, "assistant", reply)
    else:
        await update.message.reply_text("Error generating embeddings. Please try again later.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_id = update.message.message_id
    print(f"==> User uploaded an image")

    file = await context.bot.getFile(update.message.photo[-1].file_id)
    image_bytes = await file.download_as_bytearray()

    embeddings = get_image_embedding(image_bytes)
    if embeddings:
        relevant_content = await query_supabase_for_relevant_content(embeddings)
        context_text = " ".join(item['text'] for item in relevant_content)  # Adjust as needed
        reply = get_claude_response("Provide a summary of the image content.", context_text)
    detected_text = detect_text(image_bytes)
    print(f"==> Detected text: {detected_text}")
    if detected_text == '':
        await update.message.reply_text("No relevant text is detected from the image. Please try again.")
    else:
        reply: str = handle_message(user_id, detected_text)
        await update.message.reply_text(reply)
    else:
        await update.message.reply_text("Error generating embeddings. Please try again later.")

        store_message(user_id, message_id, "user", detected_text)
        store_message(user_id, message_id, "assistant", reply)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_id = update.message.message_id
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
        reply: str = handle_message(user_id, transcript)
        await update.message.reply_text(reply)
        store_message(user_id, message_id, "user", transcript)
        store_message(user_id, message_id, "assistant", reply)

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
