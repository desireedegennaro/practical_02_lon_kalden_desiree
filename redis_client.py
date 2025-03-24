import redis
import ollama
import os
import json

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

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

# Store documents in Redis
for key, document in word_docs.items():
    redis_client.hset("documents", key, document)

query = input('What is your query? ')

# Search for similar documents in Redis
all_docs = redis_client.hgetall("documents")

# Simple search: return top 3 results containing the query term
results = [key for key, doc in all_docs.items() if query.lower() in doc.lower()][:3]

documents = [all_docs[key] for key in results] if results else []

if not documents:
    print("No relevant documents found.")
else:
    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'system', 'content': doc} for doc in documents] + [
            {'role': 'user', 'content': query}
        ]
    )
    print(response['message'])
