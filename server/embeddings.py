import os
import glob
import concurrent.futures
import traceback
import logging
from openai import OpenAI, RateLimitError
from supabase import create_client, Client
from postgrest import APIError
from dotenv import load_dotenv
import backoff
from tiktoken import encoding_for_model

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set OpenAI API key
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

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


def ensure_table_exists(table_name="vectors"):
    try:
        # Check if the table exists
        supabase.table(table_name).select("id").limit(1).execute()
    except APIError as e:
        if e.code == "PGRST116":  # Table not found
            # Create the table
            supabase.postgrest.rpc("create_vectors_table").execute()
            logging.info(f"Created table: {table_name}")
        else:
            raise


def upload_embeddings_to_supabase(points, table_name="vectors"):
    ensure_table_exists(table_name)

    try:
        # Insert the points into Supabase
        data = [
            {"id": point["id"], "embedding": point["embedding"], "text": point["text"]}
            for point in points
        ]
        result = supabase.table(table_name).insert(data).execute()

        if hasattr(result, "error") and result.error:
            logging.error(f"Error uploading to Supabase: {result.error}")
        else:
            logging.info(f"Successfully uploaded {len(points)} points to Supabase")
    except Exception as e:
        logging.error(f"Error uploading to Supabase: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")


def chunk_text(text, max_tokens=512):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]


def process_file(filepath, table_name="vectors"):
    try:
        logging.info(f"Processing file: {filepath}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        file_size = os.path.getsize(filepath)
        if file_size == 0:
            logging.warning(f"File is empty: {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as file:
            sample_text = file.read()

        if not sample_text.strip():
            logging.warning(
                f"File contains no text after stripping whitespace: {filepath}"
            )
            return

        text_chunks = chunk_text(sample_text)

        points = []
        for chunk in text_chunks:
            try:
                embeddings = generate_embeddings_openai(chunk)
                if embeddings:
                    point = {
                        "id": int(hash(chunk) % 1e6),
                        "embedding": embeddings,
                        "text": chunk,
                    }
                    points.append(point)
            except Exception as e:
                logging.error(
                    f"Error generating embedding for chunk in {filepath}: {str(e)}"
                )

        if points:
            upload_embeddings_to_supabase(points, table_name)
            logging.info(
                f"Processed and uploaded {len(points)} chunks from: {filepath}"
            )
        else:
            logging.warning(f"No valid points generated for: {filepath}")
    except Exception as e:
        logging.error(f"Error processing file {filepath}: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")


def process_files_in_folder(folder_path, table_name="vectors"):
    filepaths = glob.glob(os.path.join(folder_path, "*.txt"))

    if not filepaths:
        logging.warning(f"No .txt files found in the folder: {folder_path}")
        return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, filepath, table_name)
            for filepath in filepaths
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in thread: {str(e)}")


# Set the folder path containing the .txt files
folder_path = "./scraped_content"
process_files_in_folder(folder_path)
