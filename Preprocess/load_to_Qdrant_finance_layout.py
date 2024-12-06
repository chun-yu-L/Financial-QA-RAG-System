import json
import os
from typing import Dict, List
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm


def process_json(file_path: str) -> List[tuple[Document, str]]:
    """
    Processes a JSON file from a given file path and returns a list of Document objects and their corresponding IDs.
    
    The JSON file is expected to have the following structure:
    {
        "source_id_page": "content"
    }
    
    Where "source_id_page" is a string of the format "source_id_page" and "content" is the page content.
    
    The function returns a list of tuples, where each tuple contains a Document object and its corresponding ID.
    The Document object contains the page content and metadata containing the source ID, page, category and file source.
    The ID is a randomly generated UUID.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents_and_ids = []

    for key_id, content in tqdm(
        data.items(), desc=f"Processing {os.path.basename(file_path)}"
    ):
        category = "finance"
        source_id, page = key_id.split('_')

        metadata = {
            "category": category,
            "source_id": source_id,
            "page": page,
            "file_source": "finance_layout"
        }

        doc = Document(page_content=content, metadata=metadata)

        documents_and_ids.append((doc, str(uuid4())))

    return documents_and_ids


def load_json_to_qdrant(
    folder_path: str, vector_store, batch_size: int = 50
) -> Dict[str, int]:
    """
    Load all JSON files from a folder and save them to Qdrant

    Args:
        folder_path (str): Path to the folder containing JSON files
        vector_store: Initialized Qdrant vector store
        batch_size (int): Number of documents to process at once

    Returns:
        Dict[str, int]: Dictionary with filenames and number of documents processed
    """
    all_documents = []
    all_ids = []
    files_processed = {}

    # Get all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return files_processed

    # Process each JSON file
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        try:
            docs_and_ids = process_json(file_path)
            documents, ids = zip(*docs_and_ids) if docs_and_ids else ([], [])

            all_documents.extend(documents)
            all_ids.extend(ids)

            files_processed[json_file] = len(documents)

        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            continue

    # Save all documents to Qdrant in batches
    total_batches = len(all_documents) // batch_size + (
        1 if len(all_documents) % batch_size != 0 else 0
    )

    for i in tqdm(
        range(0, len(all_documents), batch_size),
        desc="Saving to Qdrant",
        total=total_batches,
    ):
        batch_documents = all_documents[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]

        vector_store.add_documents(documents=batch_documents, ids=batch_ids)

    return files_processed


def main():
    load_dotenv()
    client = QdrantClient(url=os.getenv("qdrant_url"), timeout=60)

    ### 創一個 collection
    # 設定預設 sparse model
    client.set_sparse_model(
        embedding_model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
    )

    # 如果不存在，創建 collection
    if not client.collection_exists("finance_layout"):
        client.create_collection(
            collection_name="finance_layout",
            vectors_config=VectorParams(
                size=1024, distance=Distance.COSINE
            ),  # BAAI/bge-m3 的參數設置 (FastEmbed 沒有所以自己設)
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(),  # 前面設定的預設 sparse model 參數
            on_disk_payload=True,
        )

    ### 做一個vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="finance_layout",
        embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3",model_kwargs = {"device": "cuda"}),
        sparse_embedding=FastEmbedSparse(
            model_name="Qdrant/bm42-all-minilm-l6-v2-attentions",
            providers=["CUDAExecutionProvider"]
        ),
        sparse_vector_name="fast-sparse-bm42-all-minilm-l6-v2-attentions",
        retrieval_mode=RetrievalMode.HYBRID,
    )

    json_folder_path = "./finance_layout"
    load_json_to_qdrant(json_folder_path, vector_store, batch_size=10)


if __name__ == "__main__":
    main()
