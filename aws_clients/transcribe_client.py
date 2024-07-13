import os
import time
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

s3_client = session.client('s3')
transcribe_client = session.client('transcribe')

def transcribe_audio(file_name: str):
    extracted_text = ''

    # Upload voice message to S3
    file_path = f"temp/{file_name}"
    s3_client.upload_file(file_path, 'voicemessagebucket', file_name)

    # Start transcription job
    transcribe_client.start_transcription_job(
        TranscriptionJobName=f'{file_name}_transcription_job',
        LanguageCode='en-US',
        Media={
            'MediaFileUri': f's3://voicemessagebucket/{file_name}'
        },
        OutputBucketName='voicemessagetranscriptsbucket',
        OutputKey=f'{file_name}_transcript'
    )

    # Query for transcription job status
    # Possible transcription job statuses: QUEUED, IN_PROGRESS, FAILED, COMPLETED
    transcribe_status = transcribe_client.get_transcription_job(
        TranscriptionJobName=f'{file_name}_transcription_job'
    )['TranscriptionJob']['TranscriptionJobStatus']

    print("==> Transcribe Status: ", transcribe_status)

    while transcribe_status == 'IN_PROGRESS' or transcribe_status == 'QUEUED':
        transcribe_status = transcribe_client.get_transcription_job(
            TranscriptionJobName=f'{file_name}_transcription_job'
        )['TranscriptionJob']['TranscriptionJobStatus']
        print("==> Transcribe Status: ", transcribe_status)
        time.sleep(2)

    if transcribe_status == 'COMPLETED':
        print("==> Loading Transcript")
        # Retrieving completed transcript from s3
        transcript_file = s3_client.get_object(
            Bucket='voicemessagetranscriptsbucket', 
            Key=f'{file_name}_transcript'
            )

        transcript_content = json.loads(transcript_file['Body'].read().decode('utf-8'))
        transcript = transcript_content['results']['transcripts'][0]['transcript']
        extracted_text = transcript

    print("==> Cleaning up after transcription job")
    # Removing voice message from device
    os.remove(file_path)
    # Removing voice message from s3
    s3_client.delete_object(
        Bucket='voicemessagebucket', 
        Key=f'{file_name}'
    )
    # Removing transcript from s3
    s3_client.delete_object(
        Bucket='voicemessagetranscriptsbucket', 
        Key=f'{file_name}_transcript'
    )
    print("==> Clean up completed")
    return extracted_text
    

