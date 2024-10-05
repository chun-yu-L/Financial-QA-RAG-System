import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import chromadb
from chromadb.config import Settings


class ChromaDB:
    def __init__(self):
        self.chroma_client = self._get_chroma_client()

    def _get_chroma_client(self):
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

        return chroma_client

    def get_vector_store(
        self,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="test_collection",
    ):
        # Define the HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Setup Chroma client through Langchain
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            client=self.chroma_client,
            create_collection_if_not_exists=True,
        )


# ### testing
# chromaDB = ChromaDB()
# vector_store = chromaDB.get_vector_store()

# # # test case
# # vector_store.add_texts(
# #     texts=["This is a document about bananas", "This is a document about great tits"],
# #     metadatas = [{"id": "id1", "ref":"test_set"}, {"id": "id2", "ref":"test_set"}], # optional
# #     # ids=["id1", "id2"] # optional
# # )

# # query the vector store
# results = vector_store.similarity_search_with_relevance_scores(
#     query="This is a query document about pineapple",
#     k=4,  # how many results to return
# )
# print(results)
