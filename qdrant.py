# docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
# pip install qdrant-client

import redis
import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


qdrant = QdrantClient("localhost", port=6333)
r = redis.Redis(host='localhost', port=6379, db=0)

keys = r.keys('*')
documents = {key: r.get(key) for key in keys} 

def generate_embedding(text):
    response = ollama.embeddings(model="llama3", prompt=text)
    return response["embedding"]

#make the embedding for documents
vector_data= {key: generate_embedding(value) for key, value in documents.items()}
# create collection and put in db
if not qdrant.collection_exists(collection_name="qdrant_dbv"):
    qdrant.recreate_collection(collection_name="qdrant_dbv", vectors_config=VectorParams(size=len(vector_data[keys[0]]), distance=Distance.COSINE))

# put embeddings to qdrant now it stores them as vector embeddings
qdrant.upload_points(collection_name="qdrant_dbv",
    points=[PointStruct(id=idx,vector=vector_data[key], payload={"filename": key, "text": documents[key]} )
        for idx, key in enumerate(keys)])

query_text = "What is Redis?"
qvector = generate_embedding(query_text)
# then retrieve
search_results = qdrant.search(collection_name="qdrant_dbv",query_vector=qvector,limit=5)
for result in search_results:
    print(f"Match: {result.payload['filename']}\nText: {result.payload['text'][:200]}...\nScore: {result.score}\n")

# get response from ollama
retrieval = "\n\n".join([result.payload["text"] for result in search_results])
response = ollama.generate(model="llama3",
    prompt=f"Answer: {query_text}\n\n{retrieval}")
print("Answer:", response["response"])




