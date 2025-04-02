# practical_02_lon_kalden_desiree

# Depending on which DB you would like to use run one of the following in command prompt:
    # start redis
    docker run -d --name some-redis -p 6379:6379 redis

    # start qdrant
    docker run -d --name some-qdrant -p 6333:6333 qdrant/qdrant

    # start chromadb
    Nothing, all is done in Python!

# install dependencies
pip install redis
pip install chromadb
pip install qdrant-client
pip install ollama
pip install numpy
pip install pandas
pip install sentence-transformers
pip install tracemalloc

# to ask specific models certain questions depending on which DB you would like to use call one of the 3 following functions:
    # For Redis:
    redis_chat(query, LLM Model, word documents, embdeding model)

    # For qdrant:
    qdrant_chat(query, LLM Model, word documents, embdeding model)

    # For Chroma
    chroma_chat(query, LLM Model, word documents, embdeding model)

# to run the experiment
experiment.py