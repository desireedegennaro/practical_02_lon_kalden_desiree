import os
import redis
from qdrant_client import QdrantClient
import chromadb
from sentence_transformers import SentenceTransformer
import ollama


redis_client = redis.Redis(host="localhost", port=6379, db=0)

# chunking function
def read_data(chunk_size, overlap):
    filepath = "data/"
    word_docs = {}

    file_list = os.listdir(filepath)
    
    for file in file_list:
        with open(filepath + file, mode="r", encoding='utf-8') as infile:
            file_str = infile.read().replace("\n", " ")  # remove newlines

            prev_idx = 0
            count = 1
            while prev_idx < len(file_str):
                if prev_idx + chunk_size >= len(file_str):
                    chunk = file_str[prev_idx:]
                else:
                    chunk = file_str[prev_idx:prev_idx + chunk_size]

                chunk_key = f"{file}_chunk{count}_size{chunk_size}_overlap{overlap}"
                word_docs[chunk_key] = chunk  # store in word_docs dict
                
                # stores redis as hash with text key
                redis_client.hset(f"doc:{chunk_key}", mapping={"text": chunk})

                count += 1
                prev_idx = (prev_idx + chunk_size) - overlap

    return word_docs