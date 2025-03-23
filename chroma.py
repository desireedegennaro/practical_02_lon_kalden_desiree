# from pydantic_settings import BaseSettings
import chromadb
import ollama

# Initialize the Chroma client
# persist_directory = "C:\Users\kalde\Downloads\Northeastern\Spring 2025\DS 4300"

# Initialize the Chroma client with the persist directory
client = chromadb.Client()


# Create a collection (if it doesn't exist)
collection_name = "4300-chroma"
collection = client.create_collection(collection_name)


documents = ["This is document 1", "This is document 2", "This is document 3", 'This is a query document about florida']
ids = ["id1", "id2", "id3", 'id4']
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


response = ollama.chat(
    model='mistral',
    messages=[
        {
            'role':'system',
            'content':query_res['documents'][0][0]
        }, 
        {
            'role':'system',
            'content':query_res['documents'][0][1]
        }, 
        {
            'role':'system',
            'content':query_res['documents'][0][2]
        },
        {
            'role':'user',
            'content':query
        }
    ]
)
print(response['message'])