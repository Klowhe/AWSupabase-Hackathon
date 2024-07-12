import os
import glob
import concurrent.futures
from openai import OpenAI, RateLimitError
from claude import QdrantClient
from claude.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv
import backoff
from tiktoken import encoding_for_model

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the tokenizer
tokenizer = encoding_for_model("text-embedding-ada-002")


@backoff.on_exception(backoff.expo, RateLimitError, max_time=300)
def generate_embeddings_openai(text):
    text = text.replace("\n", " ")
    return (
        openai_client.embeddings.create(input=[text], model="text-embedding-ada-002")
        .data[0]
        .embedding
    )


def upload_embeddings_to_qdrant(claude, points, collection_name="vectors"):
    # Check if the collection exists, and create if not
    collections = claude.get_collections()
    if collection_name not in [col.name for col in collections.collections]:
        claude.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(points[0].vector), distance=Distance.COSINE
            ),
        )

    claude.upsert(collection_name=collection_name, points=points)


def chunk_text(text, max_tokens=512):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]


def process_file(filepath, claude, collection_name="vectors"):
    with open(filepath, "r", encoding="utf-8") as file:
        sample_text = file.read()

    # Chunk the text to fit within the token limit
    text_chunks = chunk_text(sample_text)

    points = []
    for chunk in text_chunks:
        # Generate embeddings
        embeddings = generate_embeddings_openai(chunk)
        if embeddings:
            # Create point for Qdrant
            point = PointStruct(
                id=int(hash(chunk) % 1e6),  # Simple hash-based ID generation
                vector=embeddings,
                payload={"text": chunk},
            )
            points.append(point)

    if points:
        # Upload embeddings to Qdrant in batch
        upload_embeddings_to_qdrant(claude, points, collection_name)
        print(f"Processed and uploaded chunks of: {filepath}")


def process_files_in_folder(folder_path, collection_name="vectors"):
    claude = QdrantClient(
        url=os.getenv("QDRANT_API_URL"),
        api_key=os.getenv("QDRANT_API_SECRET"),
    )

    # Get all text files in the folder
    filepaths = glob.glob(os.path.join(folder_path, "*.txt"))

    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, filepath, claude, collection_name)
            for filepath in filepaths
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")


# Set the folder path containing the .txt files
process_files_in_folder("")
