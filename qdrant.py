# docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
# pip install qdrant-client

import os
import ollama
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import tracemalloc

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

# Function to store embeddings in Qdrant
def store_embeddings(qdrant, collection_name, word_docs, embed_model):
    # Generate list of embeddings
    embeddings = [get_embedding(text, embed_model) for text in word_docs.values()]

    # Generate list of points from embeddings
    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"key": key, "text": word_docs[key]})
        for i, key in enumerate(word_docs.keys())
    ]
    # Insert into the collection
    qdrant.upsert(collection_name=collection_name, points=points)
    print("Embeddings stored successfully.")

def qdrant_chat(query, model, word_docs, embed_model):
    # Create qdrant client
    qdrant = QdrantClient("localhost", port=6333)

    collection_name = "qdrant_collection"
    
    # if the collection already exists, delete it to be refilled with the correectly embedded data
    try:
        qdrant.delete_collection(collection_name=collection_name)
    except Exception:
        pass
    
    # create the collection
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(get_embedding("test", embed_model)), distance=Distance.COSINE)
    )
    
    # store the embedded data int the collection
    store_embeddings(qdrant, collection_name, word_docs, embed_model)
    
    # Start tracking memory usage
    tracemalloc.start()

    # start the timer and embed the query then search for the nearest documents to compare
    start_time = time.time()
    query_embedding = get_embedding(query, embed_model)
    search_results = qdrant.search(
        collection_name=collection_name, query_vector=query_embedding, limit=3
    )

    retrieved_docs = [result.payload["text"] for result in search_results]
    
    # giving ollama the closest documents, ask it the query
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


def query_qdrant(client, query, embed_model):
    collection_name = "qdrant_collection"
    query_embedding = get_embedding(query, embed_model)

    # search
    search_results = client.search(collection_name=collection_name, query_vector=query_embedding, limit=3)
# just retrieving the docs
    retrieved_docs = [result.payload["text"] for result in search_results]
    
    return retrieved_docs


"""
# FOR TESTING
filepath = "data/"
word_docs = {}
file_list = os.listdir(filepath)
    
for file in file_list:
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

# NOTE: Available embed models include: nomic-embed-text, all-MiniLM-L6-v2, all-mpnet-base-v2
message, runtime, memory_usage = qdrant_chat(query, model="llama3.2", word_docs=word_docs, embed_model="nomic-embed-text")
print("Output:", message, "\n Runtime (s):", round(runtime, 2), "\n Maximum Memory Used:", memory_usage)
"""

