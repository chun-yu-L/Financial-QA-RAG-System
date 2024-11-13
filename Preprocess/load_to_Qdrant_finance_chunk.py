import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm

# Initialize environment and load configurations
load_dotenv()
QDRANT_URL: str = os.getenv("qdrant_url")
FILE_PATH: str = "finance_extract_directly_patched.json"
COLLECTION_NAME: str = "finance_recursive_chunk"
BATCH_SIZE: int = 1


# Initialize Text Splitter
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=2300,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )


# Load and parse documents
def load_documents(file_path: str) -> List[Document]:
    data: Dict[str, str] = json.loads(Path(file_path).read_text())
    docs: List[Document] = [
        Document(
            page_content=value,
            metadata={
                "source_id": key,
                "category": "finance",
                "file_source": file_path,
            },
        )
        for key, value in data.items()
    ]
    return docs


# Split and sequence documents
def split_and_sequence_documents(
    docs: List[Document], text_splitter: RecursiveCharacterTextSplitter
) -> List[Document]:
    split_docs: List[Document] = text_splitter.split_documents(docs)
    grouped_docs: Dict[str, List[Document]] = defaultdict(list)

    for doc in split_docs:
        grouped_docs[doc.metadata.get("source_id")].append(doc)

    for source_id, documents in grouped_docs.items():
        for seq_num, doc in enumerate(documents, start=1):
            doc.metadata["seq_num"] = seq_num

    return split_docs


# Initialize Qdrant Client and Collection
def init_qdrant_client(qdrant_url: str) -> QdrantClient:
    client = QdrantClient(url=qdrant_url, timeout=60)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
            on_disk_payload=True,
        )
    return client


# Initialize Vector Store
def get_vector_store(client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
        sparse_embedding=FastEmbedSparse(
            model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
        ),
        sparse_vector_name="fast-sparse-bm42-all-minilm-l6-v2-attentions",
        retrieval_mode=RetrievalMode.HYBRID,
    )


# Batch Upload Documents
def batch_upload_documents(
    vector_store: QdrantVectorStore, docs: List[Document], batch_size: int
) -> None:
    with tqdm(total=len(docs), desc="Uploading documents") as pbar:
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            vector_store.add_documents(batch_docs)
            pbar.update(len(batch_docs))


# Main execution flow
def main() -> None:
    text_splitter = get_text_splitter()
    docs = load_documents(FILE_PATH)
    sequenced_docs = split_and_sequence_documents(docs, text_splitter)

    client = init_qdrant_client(QDRANT_URL)
    vector_store = get_vector_store(client)

    batch_upload_documents(vector_store, sequenced_docs, BATCH_SIZE)


if __name__ == "__main__":
    main()
