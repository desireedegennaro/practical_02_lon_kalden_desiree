import time
import csv
import ollama
import chromadb
import redis
from qdrant_client import QdrantClient
from chroma import chroma_chat
from redis_driver import redis_chat 
from qdrant import qdrant_chat 
from sentence_transformers import SentenceTransformer
from Preprocess import read_data 
import pandas as pd


# embedding models to be tested
EMBEDDING_MODELS = [
    "nomic-embed-text"
    #"sentence-transformers/all-MiniLM-L6-v2",
    #"sentence-transformers/all-mpnet-base-v2"
]

# llms to be tested
LLM_MODELS = ["llama2", "mistral"]

# define chunk sizes and overlaps
#CHUNK_SIZES = [200, 500, 1000]
#OVERLAPS = [0, 50, 100]
CHUNK_SIZES = [200]
OVERLAPS = [0, 50]

#QUERY_TEXTS = ["What is redis?", "What is an AVL tree?", "How do document databases like MongoDB differ from relational databases?", "What are tradeoffs between B+ Trees and LSM?"]
QUERY_TEXTS = ["What is redis?", "What is an AVL tree?"]

# write CSV headers
EXPORT_COLS = [
        "Chunk Size", "Overlap", "Embedding Model", "Query", "Vector DB",
        "Query Time (s)", "Memory Used", "LLM Model", "LLM Response"
    ]


# query, model, word_docs, embed_model
def run_experiment(db_name):
    # loop through chunk sizes and overlaps
    export_df = pd.DataFrame(columns=EXPORT_COLS)

    if db_name == 'chroma':
        func = chroma_chat
        export_name = 'results_chroma.csv'
    elif db_name == 'qdrant':
        func = qdrant_chat
        export_name = 'results_qdrant.csv'
    elif db_name == 'redis':
        func = redis_chat
        export_name = 'results_redis.csv'

    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            print(f"\nProcessing chunk size {chunk_size}, overlap {overlap}")

            #preprocess
            word_docs = read_data(chunk_size, overlap)

            # going through each model, embedding model, and query for chunk size and overlap
            for model in LLM_MODELS:
                for embedding_model in EMBEDDING_MODELS:
                    for query in QUERY_TEXTS:
                        print(func)
                        query_result, run_time, memory = func(query, model, word_docs, embedding_model)
                        print('query:', query)
                        print('embedding_model', embedding_model)
                        print('model', model)
                        concat_df = pd.DataFrame(columns = EXPORT_COLS, data=[[chunk_size, overlap, embedding_model,
                                                                               query, db_name, run_time, memory, model, query_result]])
                        export_df = pd.concat([export_df, concat_df])
        

    export_df.to_csv(export_name)



def main():
    run_experiment('chroma')

main()
    