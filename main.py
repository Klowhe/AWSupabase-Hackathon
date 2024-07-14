# import os
# import boto3
# from typing import Final
# from dotenv import load_dotenv
# from telegram import Update
# from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
# from langchain_aws import ChatBedrock
# from langchain_core.prompts import ChatPromptTemplate

# from aws_clients.rekognition_client import detect_text
# from aws_clients.transcribe_client import transcribe_audio

# load_dotenv()
# bot_token = os.getenv('TELEGRAM_URL')

# # Create a boto3 session with the new credentials
# session = boto3.Session(
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
#     region_name=os.getenv("AWS_DEFAULT_REGION")
# )

# bedrock_client = session.client("bedrock-runtime")

# # Initialize ChatBedrock
# modelID = "anthropic.claude-v2:1"
# llm = ChatBedrock(
#     model_id=modelID,
#     client=bedrock_client,
#     model_kwargs=dict(temperature=0.9)
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a chatbot. You are in {language}."),
#         ("human", "{input}")
#     ]
# )
# bedrock_chain = prompt | llm

# # Commands
# async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     await update.message.reply_text("Hey there, I am your friendly legal assistant.")

# async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     await update.message.reply_text("You have invoked the help command.")

# # Helpers
# def handle_message(text: str):
#     # You can set this based on user preference
#     language = "english"  
#     response = bedrock_chain.invoke({'language': language, 'input': text })
#     response_content = response.content
#     return response_content

# # Responses
# async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     text: str = update.message.text
#     print(f"==> User: {text}")

#     reply: str = handle_message(text)
#     await update.message.reply_text(reply)

# async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     print(f"==> User uploaded an image")

#     # Download image file for Rekognition
#     file = await context.bot.getFile(update.message.photo[-1].file_id)
#     image_bytes = await file.download_as_bytearray()

#     detected_text = detect_text(image_bytes)
#     print(f"==> Detected text: {detected_text}")
#     if detected_text == '':
#         await update.message.reply_text("No relevant text is detected from the image. Please try again.")
#     else:
#         reply: str = handle_message(detected_text)
#         await update.message.reply_text(reply)

# async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     print(f"==> User sent a voice message")
    
#     # Download voice message for S3
#     file_id = update.message.voice.file_id
#     file = await context.bot.getFile(file_id)
#     file_name = file.file_path.split(f'https://api.telegram.org/file/bot{bot_token}/voice/', 1)[1]
#     await file.download_to_drive(custom_path=f"temp/{file_name}")

#     transcript = transcribe_audio(file_name)
#     print(f"==> Transcript: {transcript}")
#     if transcript == '':
#         await update.message.reply_text("Transcription failed. Please try again.")
#     else:
#         reply: str = handle_message(transcript)
#         await update.message.reply_text(reply)

# # Errors
# async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     print(f'Update {update} caused error {context.error}')


# if __name__ == "__main__":
#     print("=== telegram bot has been started ===")
#     app = Application.builder().token(bot_token).build()

#     # Commands
#     app.add_handler(CommandHandler('start', start_command))
#     app.add_handler(CommandHandler('help', help_command))

#     # Responses
#     app.add_handler(MessageHandler(filters.TEXT, handle_text))
#     app.add_handler(MessageHandler(filters.PHOTO, handle_image))
#     app.add_handler(MessageHandler(filters.VOICE, handle_voice))

#     # Errors
#     app.add_error_handler(error)

#     # Polls the bot
#     print("==> Polling")
#     app.run_polling(poll_interval=3)

# ___________________________________________________________________________________________________________________________

#embedded codes but does not display response in telebot

# import os
# import boto3
# from dotenv import load_dotenv
# from telegram import Update
# from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
# from openai import OpenAI, RateLimitError
# import backoff
# from langchain_aws import ChatBedrock
# from langchain_core.prompts import ChatPromptTemplate
# from aws_clients.rekognition_client import detect_text
# from aws_clients.transcribe_client import transcribe_audio
# import supabase
# from supabase import create_client, Client
# import numpy as np

# # Load environment variables
# load_dotenv()
# bot_token = os.getenv('TELEGRAM_URL')

# # Initialize OpenAI client
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Initialize AWS Rekognition client
# rekognition_client = boto3.client('rekognition')

# # Initialize Bedrock model ID and other settings
# modelID = "anthropic.claude-v2:1"
# bedrock_client = None  # Initialize later with AWS session

# # Create a boto3 session with the new credentials
# session = boto3.Session(
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
#     region_name=os.getenv("AWS_DEFAULT_REGION")
# )

# bedrock_client = session.client("bedrock-runtime")

# # Initialize ChatBedrock
# llm = ChatBedrock(
#     model_id=modelID,
#     client=bedrock_client,
#     model_kwargs=dict(temperature=0.9)
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a chatbot. You are in {language}."),
#         ("human", "{input}")
#     ]
# )
# bedrock_chain = prompt | llm

# # # Initialise Supabase client
# # supabase_client = supabase.create_client(
# #     url=os.getenv('SUPABASE_URL'),
# #     key=os.getenv('SUPABASE_KEY')
# # )
# # Initialize Supabase client
# url = os.getenv('SUPABASE_URL')
# key = os.getenv('SUPABASE_KEY')
# supabase_client: Client = create_client(url, key)

# # Commands
# async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     await update.message.reply_text("Hey there, I am your friendly legal assistant.")

# async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     await update.message.reply_text("You have invoked the help command.")

# # Helpers
# @backoff.on_exception(backoff.expo, RateLimitError, max_time=300)
# async def generate_embeddings_openai(text):
#     text = text.replace("\n", " ")
#     response = openai_client.embeddings.create(input=[text], model="text-embedding-ada-002")
#     if response and response.data and len(response.data) > 0:
#         return response.data[0].embedding
#     return None

# def find_relevant_materials(embedding):
#     # Query the Supabase database for the most similar embeddings
#     query_result = supabase_client.table('vectors').select('*').execute()
    
#     # Extract embeddings and calculate similarity
#     vectors = np.array([record['embedding'] for record in query_result['data']])
#     ids = [record['id'] for record in query_result['data']]
#     texts = [record['text'] for record in query_result['data']]

#     # Compute cosine similarities
#     similarities = np.dot(vectors, embedding) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(embedding))
#     most_similar_index = np.argmax(similarities)

#     return texts[most_similar_index] if similarities.size > 0 else "No relevant materials found."


# def handle_message(text: str):
#     # You can set this based on user preference
#     language = "english"  
#     response = bedrock_chain.invoke({'language': language, 'input': text })
#     response_content = response.content
#     return response_content

# # Responses
# async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     text: str = update.message.text
#     print(f"==> User: {text}")

#     # Step 1: Generate embeddings for user input
#     embeddings = await generate_embeddings_openai(text)

#     if embeddings:
#         print(f"==> Embeddings: {embeddings}")

#         # Step 2: Use embeddings for further processing (querying Bedrock)
#         reply: str = handle_message(text)
#         await update.message.reply_text(reply)
#     else:
#         await update.message.reply_text("Error generating embeddings. Please try again later.")

# def handle_message(text: str):
#     # You can set this based on user preference
#     language = "english"
    
#     # Generate embeddings for the input text
#     input_embedding = await generate_embeddings_openai(text)
#     if input_embedding is None:
#         return "Error generating embeddings. Please try again later."

#     # Find relevant materials from Supabase
#     relevant_material = find_relevant_materials(input_embedding)

#     # Use Bedrock for additional processing
#     response = bedrock_chain.invoke({'language': language, 'input': relevant_material })
#     response_content = response.content
#     return response_content

# # Responses
# async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     text: str = update.message.text
#     print(f"==> User: {text}")

#     reply: str = handle_message(text)
#     await update.message.reply_text(reply)

# async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     print(f"==> User uploaded an image")

#     # Download image file for Rekognition
#     file = await context.bot.getFile(update.message.photo[-1].file_id)
#     image_bytes = await file.download_as_bytearray()

#     detected_text = detect_text(image_bytes)
#     print(f"==> Detected text: {detected_text}")
#     if detected_text == '':
#         await update.message.reply_text("No relevant text is detected from the image. Please try again.")
#     else:
#         reply: str = handle_message(detected_text)
#         await update.message.reply_text(reply)

# async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     print(f"==> User sent a voice message")
    
#     # Download voice message for S3
#     file_id = update.message.voice.file_id
#     file = await context.bot.getFile(file_id)
#     file_name = file.file_path.split(f'https://api.telegram.org/file/bot{bot_token}/voice/', 1)[1]
#     await file.download_to_drive(custom_path=f"temp/{file_name}")

#     transcript = transcribe_audio(file_name)
#     print(f"==> Transcript: {transcript}")
#     if transcript == '':
#         await update.message.reply_text("Transcription failed. Please try again.")
#     else:
#         reply: str = handle_message(transcript)
#         await update.message.reply_text(reply)

# # Errors
# async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     print(f'Update {update} caused error {context.error}')


# if __name__ == "__main__":
#     print("=== telegram bot has been started ===")
#     app = Application.builder().token(bot_token).build()

#     # Commands
#     app.add_handler(CommandHandler('start', start_command))
#     app.add_handler(CommandHandler('help', help_command))

#     # Responses
#     app.add_handler(MessageHandler(filters.TEXT, handle_text))
#     app.add_handler(MessageHandler(filters.PHOTO, handle_image))
#     app.add_handler(MessageHandler(filters.VOICE, handle_voice))

#     # Errors
#     app.add_error_handler(error)

#     # Polls the bot
#     print("==> Polling")
#     app.run_polling(poll_interval=3)





# # ___________________THE LATEST VERSION WITH ASYNC ERROR___________________________________________
# import os
# import boto3
# from dotenv import load_dotenv
# from telegram import Update
# from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
# from openai import OpenAI, RateLimitError
# import backoff
# from langchain_aws import ChatBedrock
# from langchain_core.prompts import ChatPromptTemplate
# from aws_clients.rekognition_client import detect_text
# from aws_clients.transcribe_client import transcribe_audio
# from supabase import create_client, Client
# import numpy as np

# # Load environment variables
# load_dotenv()
# bot_token = os.getenv('TELEGRAM_URL')

# # Initialize OpenAI client
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Initialize AWS Rekognition client
# rekognition_client = boto3.client('rekognition')

# # Initialize Bedrock model ID and other settings
# modelID = "anthropic.claude-v2:1"
# bedrock_client = None  # Initialize later with AWS session

# # Create a boto3 session with the new credentials
# session = boto3.Session(
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
#     region_name=os.getenv("AWS_DEFAULT_REGION")
# )

# bedrock_client = session.client("bedrock-runtime")

# # Initialize ChatBedrock
# llm = ChatBedrock(
#     model_id=modelID,
#     client=bedrock_client,
#     model_kwargs=dict(temperature=0.9)
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a chatbot. You are in {language}."),
#         ("human", "{input}")
#     ]
# )
# bedrock_chain = prompt | llm

# # Initialize Supabase client
# url = os.getenv('SUPABASE_URL')
# key = os.getenv('SUPABASE_KEY')
# supabase_client: Client = create_client(url, key)

# # Commands
# async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     await update.message.reply_text("Hey there, I am your friendly legal assistant.")

# async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     await update.message.reply_text("You have invoked the help command.")

# # Helpers
# @backoff.on_exception(backoff.expo, RateLimitError, max_time=300)
# async def generate_embeddings_openai(text):
#     text = text.replace("\n", " ")
#     try:
#         response = openai_client.Embedding.create(input=[text], model="text-embedding-ada-002")
#         if response and response['data'] and len(response['data']) > 0:
#             return response['data'][0]['embedding']
#     except Exception as e:
#         print(f"Error generating embeddings: {e}")
#     return None

# async def find_relevant_materials(embedding):
#     # Query the Supabase database for the most similar embeddings
#     query_result = await supabase_client.table('vectors').select('*').execute()

#     if 'data' in query_result:
#         vectors = np.array([record['embedding'] for record in query_result['data']])
#         ids = [record['id'] for record in query_result['data']]
#         texts = [record['text'] for record in query_result['data']]

#         # Compute cosine similarities
#         similarities = np.dot(vectors, embedding) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(embedding))
#         most_similar_index = np.argmax(similarities)

#         return texts[most_similar_index] if similarities.size > 0 else "No relevant materials found."
#     return "No data found."

# async def handle_message(text: str):
#     language = "english"
    
#     # Generate embeddings for the input text
#     input_embedding = await generate_embeddings_openai(text)
#     if input_embedding is None:
#         return "Error generating embeddings. Please try again later."

#     # Find relevant materials from Supabase
#     relevant_material = find_relevant_materials(input_embedding)

#     # Use Bedrock for additional processing
#     response = bedrock_chain.invoke({'language': language, 'input': relevant_material})
#     response_content = response['content']
#     return response_content

# # Responses
# async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     text: str = update.message.text
#     print(f"==> User: {text}")

#     reply: str = handle_message(text)
#     await update.message.reply_text(reply)

# async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     print(f"==> User uploaded an image")

#     # Download image file for Rekognition
#     file = await context.bot.getFile(update.message.photo[-1].file_id)
#     image_bytes = await file.download_as_bytearray()

#     detected_text = detect_text(image_bytes)
#     print(f"==> Detected text: {detected_text}")
#     if detected_text == '':
#         await update.message.reply_text("No relevant text is detected from the image. Please try again.")
#     else:
#         reply: str = await handle_message(detected_text)
#         await update.message.reply_text(reply)

# async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     print(f"==> User sent a voice message")
    
#     # Download voice message for S3
#     file_id = update.message.voice.file_id
#     file = await context.bot.getFile(file_id)
#     file_name = file.file_path.split(f'https://api.telegram.org/file/bot{bot_token}/voice/', 1)[1]
#     await file.download_to_drive(custom_path=f"temp/{file_name}")

#     transcript = transcribe_audio(file_name)
#     print(f"==> Transcript: {transcript}")
#     if transcript == '':
#         await update.message.reply_text("Transcription failed. Please try again.")
#     else:
#         reply: str = await handle_message(transcript)
#         await update.message.reply_text(reply)

# # Errors
# async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     print(f'Update {update} caused error {context.error}')


# if __name__ == "__main__":
#     print("=== telegram bot has been started ===")
#     app = Application.builder().token(bot_token).build()

#     # Commands
#     app.add_handler(CommandHandler('start', start_command))
#     app.add_handler(CommandHandler('help', help_command))

#     # Responses
#     app.add_handler(MessageHandler(filters.TEXT, handle_text))
#     app.add_handler(MessageHandler(filters.PHOTO, handle_image))
#     app.add_handler(MessageHandler(filters.VOICE, handle_voice))

#     # Errors
#     app.add_error_handler(error)

#     # Polls the bot
#     print("==> Polling")
#     app.run_polling(poll_interval=3)
    
#     # ___________________THE LATEST VERSION WITH ASYNC ERROR___________________________________________

import os
import boto3
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI, RateLimitError
import backoff
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
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

# Initialize AWS Rekognition client
rekognition_client = boto3.client('rekognition')

# Initialize Bedrock model ID and other settings
modelID = "anthropic.claude-v2:1"
bedrock_client = None  # Initialize later with AWS session

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

def handle_message(text: str):
    # You can set this based on user preference
    language = "english"  
    response = bedrock_chain.invoke({'language': language, 'input': text })
    response_content = response.content
    return response_content

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    else:
        await update.message.reply_text("Error generating embeddings. Please try again later.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"==> User uploaded an image")

    file = await context.bot.getFile(update.message.photo[-1].file_id)
    image_bytes = await file.download_as_bytearray()

    embeddings = get_image_embedding(image_bytes)
    if embeddings:
        relevant_content = await query_supabase_for_relevant_content(embeddings)
        context_text = " ".join(item['text'] for item in relevant_content)  # Adjust as needed
        reply = get_claude_response("Provide a summary of the image content.", context_text)
        await update.message.reply_text(reply)
    else:
        await update.message.reply_text("Error generating embeddings. Please try again later.")

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
