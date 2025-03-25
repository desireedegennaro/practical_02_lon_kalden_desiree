# from pydantic_settings import BaseSettings
import chromadb
import ollama
import os

# Initialize the Chroma client
# persist_directory = "C:\Users\kalde\Downloads\Northeastern\Spring 2025\DS 4300"
# localfilepath = "C:\\Users\\budde\\Desktop\\ds4300\\"

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

query = input('what is your query?')
query_res = collection.query(
    query_texts=[query], # Chroma will embed this for you
    n_results=3 # how many results to return
)
# print(query_res)

response = ollama.chat(
    model='llama3.2',
    messages= [{'role':'system', 'content':message} for message in query_res['documents'][0]] + [
        {
            'role':'user',
            'content':query
        }
    ]
)
print(response['message'])