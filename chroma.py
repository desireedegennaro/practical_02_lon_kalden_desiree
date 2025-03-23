import chromadb
from pydantic_settings import BaseSettings

# Initialize the Chroma client
# persist_directory = "C:\Users\kalde\Downloads\Northeastern\Spring 2025\DS 4300"
client = chromadb.Client()

# Create a collection (if it doesn't exist)
collection_name = "4300-chroma"
collection = client.create_collection(collection_name)


documents = ["This is document 1", "This is document 2", "This is document 3"]
ids = ["id1", "id2", "id3"]
# metadatas = [{"source": "source1"}, {"source": "source2"}, {"source": "source3"}] # Optional metadata
collection.add(
        documents=documents,
        ids=ids,
        # metadatas=metadatas # Optional
    )

print(collection)