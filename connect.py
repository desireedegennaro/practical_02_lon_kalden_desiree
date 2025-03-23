import redis
import ollama

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
keys = r.keys('*')

for key in keys:
    content = r.get(key)
    response = ollama.generate(model='llama3',
        prompt=f"Summarize this content from {key}: {content[:2000]}"
    )
    print(f"{key} analysis:\n{response['response']}\n{'-'*50}\n")
