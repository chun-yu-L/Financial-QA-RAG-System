import chromadb
from chromadb.utils import embedding_functions


chroma_client = chromadb.HttpClient(host='localhost', port=7878)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# First time use create_collection
# collection = chroma_client.create_collection(name="my_collection", embedding_function=sentence_transformer_ef)
collection = chroma_client.get_collection(name="my_collection", embedding_function=sentence_transformer_ef)


collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["This is a query document about pineapple"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)
