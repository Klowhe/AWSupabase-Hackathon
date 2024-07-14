import json
import vecs
import boto3
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_aws import ChatBedrock, BedrockLLM
from langchain_core.prompts import ChatPromptTemplate
import boto3
import os
from dotenv import load_dotenv
from langchain_core.output_parsers.string import StrOutputParser

load_dotenv()

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)

modelID = "anthropic.claude-3-sonnet-20240229-v2:0"
Bedrock_client = boto3.client("bedrock-runtime", "us-west-2")


def add_context(sentence, db_name):

    embeddings = []
    # invoke the embeddings model for each sentence
    response = Bedrock_client.invoke_model(
        body=json.dumps({"inputText": sentence}),
        modelId="amazon.titan-embed-text-v1",
        accept="application/json",
        contentType="application/json",
    )
    # collect the embedding from the response
    response_body = json.loads(response["body"].read())
    # add the embedding to the embedding list
    embeddings.append((sentence, response_body.get("embedding"), {}))

    vx = vecs.Client(os.getenv("DB_CONNECTION"))
    sentences = vx.get_or_create_collection(name=db_name, dimension=1536)
    sentences.upsert(records=embeddings)
    sentences.create_index()


def rag_query(query_sentence, db_name, limiter):

    # create vector store client
    vx = vecs.Client(os.getenv("DB_CONNECTION"))

    # create an embedding for the query sentence
    response = Bedrock_client.invoke_model(
        body=json.dumps({"inputText": query_sentence}),
        modelId="amazon.titan-embed-text-v1",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response["body"].read())

    query_embedding = response_body.get("embedding")

    sentences = vx.get_or_create_collection(name=db_name, dimension=1536)
    # query the 'sentences' collection for the most similar sentences
    results = sentences.query(data=query_embedding, limit=limiter, include_value=True)
    # print the results
    for result in results:
        print(db_name, result)

    return results


def add_user_behaviour(sentence):

    embeddings = []

    # invoke the embeddings model for each sentence
    response = Bedrock_client.invoke_model(
        body=json.dumps({"inputText": sentence}),
        modelId="amazon.titan-embed-text-v1",
        accept="application/json",
        contentType="application/json",
    )
    # collect the embedding from the response
    response_body = json.loads(response["body"].read())
    # add the embedding to the embedding list
    embeddings.append((sentence, response_body.get("embedding"), {}))

    vx = vecs.Client(os.getenv("DB_CONNECTION"))
    sentences = vx.get_or_create_collection(name="veggy", dimension=1536)
    sentences.upsert(records=embeddings)
    sentences.create_index()
