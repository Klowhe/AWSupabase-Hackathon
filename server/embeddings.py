import os
import glob
import concurrent.futures
import traceback
import logging
import json
import boto3
from dotenv import load_dotenv
import backoff
import vecs
from transformers import RobertaTokenizer

# Load environment variables from the parent directory
load_dotenv()
# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize vecs client
try:
    vx = vecs.Client(DB_CONNECTION)
except Exception as e:
    logging.error(f"Failed to initialize vecs client: {str(e)}")
    exit(1)

# Check and handle dimension mismatch
collection_name = "vectorized_sentences_12"
desired_dimension = 1024

try:
    sentences = vx.get_collection(collection_name)
    if sentences.dimension != desired_dimension:
        logging.error(
            f"Existing collection dimension ({sentences.dimension}) does not match desired dimension ({desired_dimension})"
        )
        exit(1)
except vecs.exc.CollectionNotFound:
    logging.info(f"Collection {collection_name} not found. Creating new collection.")
    sentences = vx.get_or_create_collection(
        name=collection_name, dimension=desired_dimension
    )
except Exception as e:
    logging.error(f"Failed to get or create collection: {str(e)}")
    exit(1)

# Initialize Amazon Bedrock client
try:
    client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=AWS_DEFAULT_REGION,
    )
except Exception as e:
    logging.error(f"Failed to initialize Bedrock client: {str(e)}")
    exit(1)

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def generate_embeddings_bedrock(text):
    text = text.replace("\n", " ")

    response = client.invoke_model(
        body=json.dumps({"inputText": text}),
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json",
    )

    try:
        result = json.loads(response["body"].read())
        embedding = result.get("embedding")

        if embedding is None:
            logging.error("No embedding found in the response.")
            return None

        if len(embedding) != 1024:
            logging.warning(
                f"Unexpected embedding dimension: {len(embedding)}. Expected 1024."
            )
            return None

        logging.info(f"Generated embedding with dimension: {len(embedding)}")
        return embedding

    except Exception as e:
        logging.error(f"Failed to parse embedding from response: {str(e)}")
        logging.error(f"Response content: {response['body'].read()}")
        return None


def process_file(filepath):
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

        for chunk in sample_text:
            try:
                embedding = generate_embeddings_bedrock(chunk)
                if embedding is not None:
                    sentences.upsert(
                        records=[(sample_text, embedding, {})],
                    )
                else:
                    logging.warning(
                        f"Failed to generate valid embedding for chunk in {filepath}"
                    )
            except Exception as e:
                logging.error(
                    f"Error generating embedding for chunk in {filepath}: {str(e)}"
                )

    except Exception as e:
        logging.error(f"Error processing file {filepath}: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")


def process_files_in_folder(folder_path):
    filepaths = glob.glob(os.path.join(folder_path, "*.txt"))

    if not filepaths:
        logging.warning(f"No .txt files found in the folder: {folder_path}")
        return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, filepath) for filepath in filepaths]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in thread: {str(e)}")


if __name__ == "__main__":
    # Set the folder path containing the .txt files
    folder_path = "./scraped_content"
    process_files_in_folder(folder_path)
