from embeddings import add_context
import os
from langchain.text_splitter import NLTKTextSplitter
import nltk

directory = r"/home/chiatzeheng/Documents/projects/AWSupabase-Hackathon/scraped_content"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    # open file
    f = open(f, "r", encoding="utf-8", errors="ignore")
    data = str(f.read())

    # nltk chunking
    text_splitter = NLTKTextSplitter()
    docs = text_splitter.split_text(data)
    print(docs)

    # feed into knowledge vectordb
    for chunks in docs:
        add_context(chunks, "veggy")
        print(chunks)
