from chromadb.utils import embedding_functions
from chromadb.config import Settings
from dotenv import load_dotenv
import chromadb
import os

# get user & pwd from env
load_dotenv()
chromadb_host = os.getenv('chromadb_host')
chromadb_user = os.getenv('chromadb_user')
chromadb_pwd = os.getenv('chromadb_password')

# DB connection using base auth
chroma_client = chromadb.HttpClient(
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
        chroma_client_auth_credentials= f'{chromadb_user}:{chromadb_pwd}'
        ),
    host=chromadb_host, port=7878
)

# if everything is correctly configured the below should list all collections
if chroma_client.list_collections():
    print('connected')
else:
    print('connection error')



# define the embedding model
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# First time use create_collection
# collection = chroma_client.create_collection(name="test_collection", embedding_function=sentence_transformer_ef)
collection = chroma_client.get_collection(name="test_collection", embedding_function=sentence_transformer_ef)

# test case
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
