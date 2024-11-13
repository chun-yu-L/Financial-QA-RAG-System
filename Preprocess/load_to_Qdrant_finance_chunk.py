"""
This module handles the process of loading, splitting, and uploading financial documents to Qdrant, a vector database,
to facilitate document retrieval and search.

Functions:
    - get_text_splitter: Creates a text splitter that recursively splits text into chunks by specified separators.
    - load_documents: Loads and parses a JSON file into a list of Document objects.
    - split_and_sequence_documents: Splits documents into chunks and assigns sequence numbers.
    - init_qdrant_client: Initializes a Qdrant client and creates a collection if it doesn't exist.
    - get_vector_store: Sets up a QdrantVectorStore with specific embedding configurations.
    - batch_upload_documents: Uploads documents to Qdrant in batches, with progress tracking.
    - main: Orchestrates the full process from loading documents, splitting, and batch uploading to Qdrant.

Environment Variables:
    - QDRANT_URL: The URL of the Qdrant instance.
    - FILE_PATH: The path to the JSON file containing the documents.
    - BATCH_SIZE: The number of documents to upload in each batch.

Usage:
    Run this module directly to execute the main function, which performs the entire data processing and uploading
    workflow.
"""

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
    """
    Returns an instance of RecursiveCharacterTextSplitter that splits text into chunks
    by recursive splitting on separators.

    The chunk size is set to 2300 and the chunk overlap is set to 200. This means
    that the generated chunks will not exceed 2300 characters and that any given
    character will be present in two chunks at most.

    The length of a chunk is determined by the number of characters in the chunk.
    The is_separator_regex parameter is set to False, which means that the separators
    are treated as literal strings rather than regular expressions.
    """
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
    """
    Loads JSON file and parses it into a list of Document objects.

    The JSON file is expected to contain a dictionary where the keys are source IDs
    and the values are the text content of the documents.

    Args:
        file_path (str): Path to JSON file.

    Returns:
        List[Document]: List of Document objects.
    """
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
    """
    Splits the given documents into chunks and assigns a sequence number to each chunk within a document.

    Args:
        docs (List[Document]): The list of documents to split and sequence.
        text_splitter (RecursiveCharacterTextSplitter): The text splitter used to split the documents.

    Returns:
        List[Document]: The list of split and sequenced documents.
    """
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
    """
    Initialize a Qdrant client and create a collection if it doesn't exist.

    This function sets up a QdrantClient instance with a specified URL and
    a timeout of 60 seconds. It checks whether a collection with the name
    defined in COLLECTION_NAME exists. If not, it creates the collection
    with specified vector parameters, including vector size, distance metric,
    and sparse vector configuration. The collection is configured to store
    payloads on disk.

    Args:
        qdrant_url (str): The URL of the Qdrant service.

    Returns:
        QdrantClient: An instance of QdrantClient connected to the specified URL.
    """

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
    """
    Initialize and return a QdrantVectorStore instance.

    Args:
        client (QdrantClient): Qdrant client to use for vector store operations.

    Returns:
        QdrantVectorStore: Qdrant vector store instance.
    """
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
    """Upload documents to Qdrant in batches.

    Uploads a list of documents to Qdrant in batches, with a progress bar.

    Args:
        vector_store: The QdrantVectorStore to upload to.
        docs: The list of documents to upload.
        batch_size: The number of documents to upload in each batch.
    """
    with tqdm(total=len(docs), desc="Uploading documents") as pbar:
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            vector_store.add_documents(batch_docs)
            pbar.update(len(batch_docs))


# Main execution flow
def main() -> None:
    """
    Main execution flow for loading documents, splitting them into chunks,
    and uploading the chunks to Qdrant.

    This function assumes that the QDRANT_URL, FILE_PATH, and BATCH_SIZE
    variables are set in the environment.

    It works by first loading the documents from the file at FILE_PATH,
    then splitting them into chunks using a text splitter. The chunks are
    then uploaded to Qdrant using a vector store, in batches of BATCH_SIZE.
    """
    text_splitter = get_text_splitter()
    docs = load_documents(FILE_PATH)
    sequenced_docs = split_and_sequence_documents(docs, text_splitter)

    client = init_qdrant_client(QDRANT_URL)
    vector_store = get_vector_store(client)

    batch_upload_documents(vector_store, sequenced_docs, BATCH_SIZE)


if __name__ == "__main__":
    main()
