# from pydantic_settings import BaseSettings
import chromadb
import ollama
import os
import time

# Initialize the Chroma client
# persist_directory = "C:\Users\kalde\Downloads\Northeastern\Spring 2025\DS 4300"
# localfilepath = "C:\\Users\\budde\\Desktop\\ds4300\\"

query = input('what is your query?')

# print(query_res)
def chroma_chat(query, model, word_docs):
    # Initialize the Chroma client with the persist directory
    client = chromadb.Client()

    # Create a collection (if it doesn't exist)
    collection_name = "4300-chroma"
    collection = client.create_collection(collection_name)

    documents = list(word_docs.values())
    ids = list(word_docs.keys())
    # metadatas = [{"source": "source1"}, {"source": "source2"}, {"source": "source3"}] # Optional metadata
    collection.add(
            documents=documents,
            ids=ids,
            # metadatas=metadatas # Optional
        )
    start_time = time.time
    query_res = collection.query(
    query_texts=[query], # Chroma will embed this for you
    n_results=3 # how many results to return
)
    response = ollama.chat(
        model=model,
        messages= [{'role':'system', 'content':message} for message in query_res['documents'][0]] + [
            {
                'role':'user',
                'content':query
            }
        ]
    )

    return response['message'], time.time - start_time