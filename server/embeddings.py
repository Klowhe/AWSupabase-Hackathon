import os
import requests
from supabase import create_client, Client
import voyageai
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.environ.get("DATABASE_URL")


# Function to chunk text
def chunk_text(text, max_chunk_size=1000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = " ".join(words[i : i + max_chunk_size])
        chunks.append(chunk)
    return chunks


# Function to get embeddings using Claude (OpenAI in this example)
def get_embedding(text):
    result = vo.embed(text, model="voyage-2", input_type="document")
    return result.embeddings[0]


# Function to upload embeddings to Supabase
def upload_to_supabase(supabase_url, embeddings_data):
    supabase: Client = create_client(supabase_url)
    data, count = supabase.table("embeddings").insert(embeddings_data).execute()
    return data, count


# Iterate through the files in the scraped_content directory
directory = "scraped_content"
embeddings_data = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
            chunks = chunk_text(content)
            for chunk in chunks:
                embedding = get_embedding(chunk)
                embeddings_data.append(
                    {"file_name": filename, "chunk": chunk, "embedding": embedding}
                )

# Upload embeddings to Supabase
upload_to_supabase(supabase_url, embeddings_data)
print("Upload completed.")
