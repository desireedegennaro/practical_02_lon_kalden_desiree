import chromadb
import ollama
import time
import os
from sentence_transformers import SentenceTransformer
import tracemalloc
from Preprocess import read_data


# Function to generate embeddings
# NOTE: only works with nomic-embed-text and models from Sentence Transformer
def get_embedding(text: str, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> list:
    if model == "nomic-embed-text":
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    else:
        nmodel = SentenceTransformer(model)
        return nmodel.encode(text).tolist()


def store_embeddings(collection, word_docs, embed_model):
    """Store document embeddings in ChromaDB."""
    # collect data necessary for adding to the collection
    documents = list(word_docs.values())
    ids = list(word_docs.keys())
    texts = list(word_docs.values())
    embeddings = [get_embedding(text, embed_model) for text in word_docs.values()]
  
    # add to the collection
    collection.add(
        documents=texts,
        ids=ids,
        embeddings=embeddings
    )
    print("Embeddings stored successfully.")

def chroma_chat(query, model, word_docs, embed_model):
    # Initialize Chroma client
    client = chromadb.PersistentClient(path="./chroma_db")  # Persist the data

    collection_name = "4300-chroma"

    # if the collection already exists delete it and recreate it to ensure all data is embedded the same
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass 

    # create the collection
    collection = client.get_or_create_collection(name=collection_name)

    # Store embeddings if they haven't been added
    if collection.count() == 0:
        store_embeddings(collection, word_docs, embed_model)

    # Start tracking memory usage
    tracemalloc.start()

    # start the timer and embed the query 
    start_time = time.time()

    query_embedding = get_embedding(query, embed_model)

    # locate the 3 closest documents
    query_res = collection.query(
        query_embeddings=[query_embedding], 
        n_results=3
    )

    retrieved_docs = query_res["documents"][0] if query_res["documents"] else []

    # feed the documents to the ollama model to generate a response to the users query
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

def query_chroma(client, query, embed_model):
    collection_name = "4300-chroma"
    collection = client.get_or_create_collection(name=collection_name)

    query_embedding = get_embedding(query, embed_model)
    # search chroma
    query_res = collection.query(query_embeddings=[query_embedding], n_results=3)
    # just get the docs
    retrieved_docs = query_res["documents"][0] if query_res["documents"] else []
    
    return retrieved_docs, query_res["distances"][0] if query_res["documents"] else []
