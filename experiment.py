import time
import csv
import ollama
import chromadb
import redis
from qdrant_client import QdrantClient
from chroma import store_embeddings as store_chroma, query_chroma
from redis_driver import store_embedding as store_redis, query_redis
from qdrant import store_embeddings as store_qdrant, query_qdrant
from sentence_transformers import SentenceTransformer
from Preprocess import read_data 

# make clients for vector DBs
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="4300-chroma")
redis_client = redis.Redis(host="localhost", port=6379, db=0)
qdrant_client = QdrantClient("localhost", port=6333)

# embedding models to be tested
embedding_models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "hkunlp/instructor-xl"
]

# llms to be tested
llm_models = ["llama2", "mistral"]

# define chunk sizes and overlaps
chunk_sizes = [200, 500, 1000]
overlaps = [0, 50, 100]

query_texts = ["What is redis?", "What is an AVL tree?", "How do document databases like MongoDB differ from relational databases?", "What are tradeoffs between B+ Trees and LSM?"]

# path to save results
csv_filename = "experiment_results.csv"

# write CSV headers
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Chunk Size", "Overlap", "Embedding Model", "Query", "Vector DB",
        "Query Time (s)", "Retrieved Docs", "LLM Model", "LLM Response"
    ])

# get embedding for the texts
def get_embedding(text, embed_model):
    model = SentenceTransformer(embed_model)
    return model.encode(text).tolist()

def run_experiment():
    # loop through chunk sizes and overlaps
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            print(f"\nProcessing chunk size {chunk_size}, overlap {overlap}")

            # preprocess documents
            #word_docs = read_data(chunk_sizes=[chunk_size], overlaps=[overlap])
            # Load document chunks
            word_docs = read_data(chunk_sizes=[chunk_size], overlaps=[overlap])

            # check if embedded in redis already so dont need to store each time
            existing_keys = set(redis_client.keys("doc:*")) 
            # already stored
            new_word_docs = {k: v for k, v in word_docs.items() if f"doc:{k}" not in existing_keys}
            if not new_word_docs:
                print("Docs embedded already.")
            else:
                print(f"{len(new_word_docs)} left.")
                word_docs = new_word_docs # only keeping the new docs


            for embed_model in embedding_models:
                print(f"\nEmbedding model: {embed_model}")

                # store embeddings in each database
                # for redis the parameters dont match so we need to loop over word docs
                DOC_PREFIX = "doc:"
                for key, text in word_docs.items():
                    clean_text = " ".join(text) if isinstance(text, list) else str(text) 
                    embedding = get_embedding(clean_text, embed_model)
                    store_redis(key, clean_text, embedding, DOC_PREFIX, redis_client)

                store_chroma(chroma_client, word_docs, embed_model)
                store_qdrant(qdrant_client, "qdrant_collection", word_docs, embed_model)

                # run queries
                for query_text in query_texts:
                    print(f"\nQuerying: {query_text}")

                    redis_docs = query_redis(redis_client, query_text, embed_model)
                    chroma_docs = query_chroma(chroma_client, query_text, embed_model)
                    qdrant_docs = query_qdrant(qdrant_client, query_text, embed_model)

                    print(f"Redis time: {redis_time:.2f}s, docs: {redis_docs}")
                    print(f"Chroma time: {chroma_time:.2f}s, docs: {chroma_docs}")
                    print(f"Qdrant time: {qdrant_time:.2f}s, docs: {qdrant_docs}")

                    # send top documents to LLMs
                    for db_name, docs, query_time in results:
                        for llm in llm_models:
                            print(f"\nQuerying LLM: {llm}")

                            response = ollama.chat(
                                model=llm,
                                messages=[
                                    {"role": "system", "content": doc} for doc in chroma_docs
                                ] + [{"role": "user", "content": query_text}]
                            )
                            # write to csv
                            with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
                                writer = csv.writer(file)
                                writer.writerow([
                                    chunk_size, overlap, embed_model, query_text, db_name, query_time, "; ".join(docs), llm, response['message'][:200]])

                            print(f"{llm} response: {response['message'][:200]}...")

# run experiment
if __name__ == "__main__":
    run_experiment()
