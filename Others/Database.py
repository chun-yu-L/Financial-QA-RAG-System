import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import chromadb
from chromadb.config import Settings


class ChromaDB:
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(self):
        self.chroma_client = self._get_chroma_client()
        self.current_collection = None
        self.embeddings = None

    def _get_chroma_client(self):
        load_dotenv()
        chromadb_host = os.getenv("chromadb_host")
        chromadb_user = os.getenv("chromadb_user")
        chromadb_pwd = os.getenv("chromadb_password")

        chroma_client = chromadb.HttpClient(
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
                chroma_client_auth_credentials=f"{chromadb_user}:{chromadb_pwd}",
            ),
            host=chromadb_host,
            port=7878,
        )

        if chroma_client.list_collections():
            print("Connected to ChromaDB")
        else:
            print("Connection error")

        return chroma_client

    def get_vector_store(
        self,
        collection_name: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> Chroma:
        """Initialize or get a vector store with specified embedding model"""
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            client=self.chroma_client,
            collection_metadata={"hf_model": embedding_model},
            create_collection_if_not_exists=True,
        )
        
        self.current_collection = vector_store
        return vector_store
