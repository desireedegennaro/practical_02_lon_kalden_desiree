import os
import time
import tracemalloc
import redis
import csv
from redis_driver import redis_chat  
from Preprocess import read_data
from qdrant import qdrant_chat
from chroma import chroma_chat

redis_client = redis.Redis(host="localhost", port=6379, db=0)

VECTOR_DB_MODELS = {"chroma": chroma_chat, "qdrant": qdrant_chat, "redis": redis_chat}

if __name__ == "__main__":
    word_docs = read_data()    
    model_choice = input("Choose model (chroma, qdrant, redis): ").strip().lower()
    model_func = VECTOR_DB_MODELS.get(model_choice)

    if model_func is None:
        print(f"choose from: chroma, qdrant, redis.")
        exit()

    r = redis.Redis(host='localhost', port=6379, db=0)
    keys = r.keys('*')
    documents = {}

    for key in keys:
        key_decoded = key.decode()
        key_type = r.type(key).decode()  

        value = redis_client.hget(key, "text")
        if value is not None:
            try:
                documents[key_decoded] = value.decode()
            except Exception as e:
                print(f"⚠️ Error decoding key '{key_decoded}': {e}")


    query = input("What is your query? ").strip()
# track time
    tracemalloc.start()
    start_time = time.time()

    message, runtime, memory_usage = model_func(query, model="llama3.2", word_docs=word_docs, embed_model="nomic-embed-text")

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("Model Used:", model_choice)
    print("Response:", message)
    print("Runtime (s):", round(runtime, 2))
    print("Memory Used:", peak_memory, "bytes")
