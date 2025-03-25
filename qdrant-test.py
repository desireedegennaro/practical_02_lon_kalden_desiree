# docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
# pip install qdrant-client

import os
import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Load data from files
filepath = "data\\"
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

# Initialize Qdrant client
qdrant = QdrantClient("localhost", port=6333)

def generate_embedding(text):
    response = ollama.embeddings(model="llama3.2", prompt=text)
    return response["embedding"]

# Generate embeddings for documents
vector_data = {key: generate_embedding(value) for key, value in word_docs.items()}

# Create collection in Qdrant if it doesn't exist
if not qdrant.collection_exists(collection_name="qdrant_dbv"):
    qdrant.recreate_collection(
        collection_name="qdrant_dbv",
        vectors_config=VectorParams(size=len(vector_data[list(vector_data.keys())[0]]), distance=Distance.COSINE)
    )

# Upload embeddings to Qdrant
to_upload = [
    PointStruct(id=idx, vector=vector_data[key], payload={"filename": key, "text": word_docs[key]})
    for idx, key in enumerate(word_docs.keys())
]
qdrant.upload_points(collection_name="qdrant_dbv", points=to_upload)

# Query and search
query_text = "What is Redis?"
qvector = generate_embedding(query_text)
search_results = qdrant.search(collection_name="qdrant_dbv", query_vector=qvector)

for result in search_results:
    print(f"Match: {result.payload['filename']}\nText: {result.payload['text'][:200]}...\nScore: {result.score}\n")

# Generate response with Ollama
retrieval = "\n\n".join([result.payload["text"] for result in search_results])
response = ollama.generate(model="llama3.2", prompt=f"Answer: {query_text}\n\n{retrieval}")
print("Answer:", response["response"])