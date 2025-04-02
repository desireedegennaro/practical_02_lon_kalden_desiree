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
    "nomic-embed-text",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

# llms to be tested
LLM_MODELS = ["llama2", "mistral"]

# define chunk sizes and overlaps
CHUNK_SIZES = [200, 500, 1000]
OVERLAPS = [0, 50, 100]

QUERY_TEXTS = ["What is redis?", "What is an AVL tree?", "How do document databases like MongoDB differ from relational databases?", "What are tradeoffs between B+ Trees and LSM?"]

# write CSV headers
EXPORT_COLS = [
        "Chunk Size", "Overlap", "Embedding Model", "Query", "Vector DB",
        "Query Time (s)", "Memory Used", "LLM Model", "LLM Response"
    ]


# query, model, word_docs, embed_model
def run_experiment(db_name, embedding_models, llm_models, chunk_sizes, overlaps, query_texts, export_name):
    # loop through chunk sizes and overlaps
    export_df = pd.DataFrame(columns=EXPORT_COLS)

    if db_name == 'chroma':
        func = chroma_chat
    elif db_name == 'qdrant':
        func = qdrant_chat
    elif db_name == 'redis':
        func = redis_chat

    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            print(f"\nProcessing chunk size {chunk_size}, overlap {overlap}")

            #preprocess
            word_docs = read_data(chunk_size, overlap)

            # going through each model, embedding model, and query for chunk size and overlap
            for model in llm_models:
                for embedding_model in embedding_models:
                    for query in query_texts:
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
    # DIRECTIONS: uncomment out the database you want to test, set up the container if redis or qdrant, and run file!
    #run_experiment('chroma', EMBEDDING_MODELS, LLM_MODELS, [200], [0], QUERY_TEXTS, 'results_chroma.csv')
    #run_experiment('qdrant', EMBEDDING_MODELS, LLM_MODELS, [200], [0], QUERY_TEXTS, 'results_qdrant.csv')
    #run_experiment('redis', EMBEDDING_MODELS, LLM_MODELS, [200], [0], QUERY_TEXTS, 'results_redis.csv')
    best_embed = ["nomic-embed-text"]
    best_model = ["mistral"]
    queries = ["What is Redis?", "What are the benefits of an AVL Tree?"]
    run_experiment('chroma', best_embed, best_model, CHUNK_SIZES, OVERLAPS, queries, 'chroma_chunk_results.csv')
    #run_experiment('chroma', best_embed, best_model, CHUNK_SIZES, OVERLAPS, queries, 'qdrant_chunk_results.csv')
    #run_experiment('chroma', best_embed, best_model, CHUNK_SIZES, OVERLAPS, queries, 'redis_chunk_results.csv')
main()
    