import ollama
import redis
import numpy as np
import os
from redis.commands.search.query import Query

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Create an index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass
    
    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Store the calculated embedding in Redis
def store_embedding(doc_id: str, text: str, embedding: list):
    key = f"{DOC_PREFIX}{doc_id}"
    redis_client.hset(
        key,
        mapping={
            "text": text,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {text}")

if __name__ == "__main__":
    create_hnsw_index()

    # Load text files
    filepath = "data/"
    word_docs = {}
    file_list = os.listdir(filepath)
    
    for file in file_list:
        words = []
        with open(filepath + file, mode="r", encoding='utf-8') as infile:
            key = '1'
            for line in infile.readlines():
                line = line.strip()
                if line.isnumeric():
                    key = file + ' slide ' + line
                    word_docs[key] = ''
                
                if key in word_docs:
                    word_docs[key] += line
                else:
                    word_docs[key] = ''

    # Store embeddings in Redis
    for i, (key, text) in enumerate(word_docs.items()):
        embedding = get_embedding(text)
        store_embedding(key, text, embedding)

    # Query the database
    query_text = input("What is your query? ")
    embedding = get_embedding(query_text)
    
    q = (
        Query("*=>[KNN 3 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("text", "vector_distance")
        .dialect(2)
    )
    
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    
    retrieved_docs = [doc.text for doc in res.docs]
    
    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'system', 'content': doc} for doc in retrieved_docs] + [
            {'role': 'user', 'content': query_text}
        ]
    )
    print(response['message'])
