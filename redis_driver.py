import redis
import ollama
import os
import json
import time


query = input('What is your query? ')


def redis_chat(query, model, word_docs):
    # Initialize Redis client
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    # Store documents in Redis
    for key, document in word_docs.items():
        redis_client.hset("documents", key, document)

    start_time = time.time()
    # Search for similar documents in Redis
    all_docs = redis_client.hgetall("documents")

    # Simple search: return top 3 results containing the query term
    results = [key for key, doc in all_docs.items() if query.lower() in doc.lower()][:3]

    documents = [all_docs[key] for key in results] if results else []

    response = ollama.chat(
        model=model,
        messages=[{'role': 'system', 'content': doc} for doc in documents] + [
            {'role': 'user', 'content': query}
        ]
    )
    return response['message'], (time.time() - start_time)


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
message, runtime = redis_chat(query, model="llama3.2", word_docs=word_docs)
print(message, runtime)