import redis
import ollama
import os
import time
import numpy as np
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import tracemalloc

# Create an index in Redis (from sample code)
def create_hnsw_index(redis_client, INDEX_NAME, DOC_PREFIX, VECTOR_DIM, DISTANCE_METRIC):
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

# Function to generate embeddings
# NOTE: only works with nomic-embed-text and models from Sentence Transformer
def get_embedding(text: str, model: str = "all-MiniLM-L6-v2") -> list:
    # nomic-embed-text can be done through ollama where as the others must be done usign Sentence Transformer
    if model == "nomic-embed-text":
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    else:
        nmodel = SentenceTransformer(model)
        return nmodel.encode(text).tolist()

# Store the calculated embedding in Redis (from sample code)
def store_embedding(doc_id: str, text: str, embedding: list, DOC_PREFIX, redis_client):
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

def redis_chat(query, model, word_docs, embed_model):
    # Initialize Redis connection
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    # define global variables (from sample code)
    VECTOR_DIM = 768
    INDEX_NAME = "embedding_index"
    DOC_PREFIX = "doc:"
    DISTANCE_METRIC = "COSINE"

    create_hnsw_index(redis_client, INDEX_NAME, DOC_PREFIX, VECTOR_DIM, DISTANCE_METRIC)

    # Store embeddings in Redis (from sample code)
    for i, (key, text) in enumerate(word_docs.items()):
        embedding = get_embedding(text, embed_model)
        store_embedding(key, text, embedding, DOC_PREFIX, redis_client)
    # Start tracking memory usage
    tracemalloc.start()

    # begin timer and embed/store the query
    start_time = time.time()
    embedding = get_embedding(query, embed_model)
    
    q = (
        Query("*=>[KNN 3 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("text", "vector_distance")
        .dialect(2)
    )
    
    # search for the closest documents
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    
    retrieved_docs = [doc.text for doc in res.docs]
    
    # given the closest documents ask ollama the given query
    response = ollama.chat(
        model=model,
        messages=[{'role': 'system', 'content': doc} for doc in retrieved_docs] + [
            {'role': 'user', 'content': query}
        ]
    )

    # Get memory usage statistics
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return response['message'], (time.time() - start_time), peak_memory



