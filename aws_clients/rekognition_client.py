import os
import boto3
from dotenv import load_dotenv

load_dotenv()

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

rekognition_client = session.client('rekognition')

def detect_text(image_bytes):
    print("==> Detecting text from image with AWS Rekognition")
    response = rekognition_client.detect_text(Image={'Bytes': image_bytes})

    detected_text = []
    # hacky method to fix duplicated detected text
    halfway_point = (len(response['TextDetections']) // 2) - 1
    for text_detection in response['TextDetections'][:halfway_point]:
        detected_text.append(text_detection['DetectedText'])
    return ' '.join(detected_text)