import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import chromadb
from chromadb.config import Settings

# get user & pwd from env
load_dotenv()
chromadb_host = os.getenv("chromadb_host")
chromadb_user = os.getenv("chromadb_user")
chromadb_pwd = os.getenv("chromadb_password")

# DB connection using base auth
chroma_client = chromadb.HttpClient(
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
        chroma_client_auth_credentials=f"{chromadb_user}:{chromadb_pwd}",
    ),
    host=chromadb_host,
    port=7878,
)

# if everything is correctly configured the below should list all collections
if chroma_client.list_collections():
    print("connected")
else:
    print("connection error")

# Define the HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Setup Chroma client through Langchain
vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings,
    client=chroma_client,
    create_collection_if_not_exists=True,
)

# # test case
# vector_store.add_texts(
#     texts=["This is a document about bananas", "This is a document about great tits"],
#     metadatas = [{"id": "id1", "ref":"test_set"}, {"id": "id2", "ref":"test_set"}], # optional
#     # ids=["id1", "id2"] # optional
# )

# query the vector store
results = vector_store.similarity_search_with_relevance_scores(
    query="This is a query document about pineapple",
    k=4,  # how many results to return
)
print(results)
