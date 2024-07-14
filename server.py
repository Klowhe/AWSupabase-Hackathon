
import os
import boto3
import json
import vecs
import backoff
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from supabase import create_client, Client
import uvicorn

from aws_clients.rekognition_client import detect_text
from aws_clients.transcribe_client import transcribe_audio

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

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


class Message(BaseModel):
    user_id: str
    content: str
    type: str  # 'text', 'image', or 'voice'


def store_message(user_id, message_id, role, content):
    (
        supabase_client.table("chat_messages")
        .insert(
            {
                "user_id": user_id,
                "message_id": message_id,
                "role": role,
                "content": content,
                "session_id": user_sessions.get(user_id, "default"),
            }
        )
        .execute()
    )


def retrieve_chat_history(user_id):
    response = (
        supabase_client.table("chat_messages")
        .select("role, content")
        .eq("user_id", user_id)
        .eq("session_id", user_sessions.get(user_id, "default"))
        .order("message_id", desc=False)
        .execute()
    )

    chat_history = response.data
    return chat_history


async def generate_embeddings_openai(text: str):
    # Implement your embedding generation logic here
    pass


async def handle_message(user_id: str, text: str):
    embeddings = await generate_embeddings_openai(text)
    if embeddings:
        rag_results = sentences.query(data=embeddings, limit=3, include_value=True)
        print("rag results: ", rag_results)
        chat_history = retrieve_chat_history(user_id)
        response = bedrock_chain.invoke(
            {"history": chat_history, "input": text, "context": rag_results}
        )
        response_content = response.content
        return response_content
    else:
        return "Error generating embeddings. Please try again later."


@app.post("/start")
async def start_command(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    message_id = data.get("message_id")
    user_sessions[user_id] = message_id
    return {"message": "Hey there, I am your friendly legal assistant."}


@app.post("/message")
async def handle_message_endpoint(message: Message):
    user_id = message.user_id
    content = message.content
    message_type = message.type

    if message_type == "text":
        reply = await handle_message(user_id, content)
    elif message_type == "image":
        detected_text = detect_text(content)  # Assuming content is image bytes
        if detected_text:
            reply = await handle_message(user_id, detected_text)
        else:
            reply = "No relevant text is detected from the image. Please try again."
    elif message_type == "voice":
        transcript = transcribe_audio(content)  # Assuming content is audio file path
        if transcript:
            reply = await handle_message(user_id, transcript)
        else:
            reply = "Transcription failed. Please try again."
    else:
        raise HTTPException(status_code=400, detail="Invalid message type")

    # Store the message and reply
    store_message(user_id, user_sessions.get(user_id, "default"), "user", content)
    store_message(user_id, user_sessions.get(user_id, "default"), "assistant", reply)

    return {"reply": reply}


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
