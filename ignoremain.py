import os
import vecs
import boto3
import json

# Initialize the Bedrock runtime client
client = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)

# Create a vecs client
vx = vecs.create_client(os.getenv("DB_CONNECTION"))

collection_name = "sentences"
# The query sentence
query_sentence = "A quick animal jumps over a lazy one."

# Invoke the model to get the embedding
response = client.invoke_model(
    body=json.dumps({"inputText": query_sentence}),
    modelId="amazon.titan-embed-text-v2:0",
    accept="application/json",
    contentType="application/json",
)

sentences = vx.get_collection(name="sentences")

# Process the response to extract the embedding
response_body = json.loads(response["body"].read())
query_embedding = response_body.get("embedding")

# Query the 'sentences' collection for the most similar sentences
results = sentences.query(data=query_embedding, limit=3, include_value=True)

# Print the results
for result in results:
    print(result)
