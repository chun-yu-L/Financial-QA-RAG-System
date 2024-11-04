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


# Save title and sequence_number to metadata
# Input json from the output of split_insurance_section.py
def process_insurance_chunk_json(file_path: str) -> List[tuple[Document, str]]:
    '''
    Sample
    metadata: {
        "category":"insurance"
        "source_id":"260"
        "title":"第 十 八 條"
        "sequence_number":7
        "file_source":"insurance_chunk.json"
    }
    '''
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents_and_ids = []

    for source_id, content in tqdm(
        data.items(), desc=f"Processing {os.path.basename(file_path)}"
    ):
        metadata = content.get('metadata', {})
        sections = content.get('sections', [])

        for section in sections:
            title = section.get('title', 0)
            sequence_number = section.get('sequence_number', 0)
            section_content = section.get('content','')
            
            section_metadata = {
                **metadata,
                'title': title,
                'sequence_number': sequence_number,
                'file_source': os.path.basename(file_path)
            }
            
            doc = Document(
                page_content=section_content,
                metadata=section_metadata
            )
            
            documents_and_ids.append((doc, str(uuid4())))

    return documents_and_ids

def load_json_to_qdrant(
    folder_path: str, vector_store, batch_size: int = 50
) -> Dict[str, int]:
    """
    Load all JSON files from a folder and save them to ChromaDB

    Args:
        folder_path (str): Path to the folder containing JSON files
        vector_store (Chroma): Initialized ChromaDB vector store
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
            docs_and_ids = process_insurance_chunk_json(file_path)
            documents, ids = zip(*docs_and_ids) if docs_and_ids else ([], [])

            all_documents.extend(documents)
            all_ids.extend(ids)

            files_processed[json_file] = len(documents)

        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            continue

    # Save all documents to ChromaDB in batches
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
    if not client.collection_exists("insurance_chunk"):
        client.create_collection(
            collection_name="insurance_chunk",
            vectors_config=VectorParams(
                size=1024, distance=Distance.COSINE
            ),  # BAAI/bge-m3 的參數設置 (FastEmbed 沒有所以自己設)
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(),  # 前面設定的預設 sparse model 參數
            on_disk_payload=True,
        )

    ### 做一個vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="insurance_chunk",
        embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
        sparse_embedding=FastEmbedSparse(
            model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
        ),
        sparse_vector_name="fast-sparse-bm42-all-minilm-l6-v2-attentions",
        retrieval_mode=RetrievalMode.HYBRID,
    )

    json_folder_path = "./chunk_json"
    load_json_to_qdrant(json_folder_path, vector_store, batch_size=50)


if __name__ == "__main__":
    main()
