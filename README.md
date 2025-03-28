# practical_02_lon_kalden_desiree

# start redis
docker run -d --name some-redis -p 6379:6379 redis

# start qdrant
docker run -d --name some-qdrant -p 6333:6333 qdrant/qdrant

# start chromadb
docker run -d --name chromadb -p 8000:8000 chromadb/chromadb

# to ask specific models any questions
python main.py

# to run the experiment
experiment.py