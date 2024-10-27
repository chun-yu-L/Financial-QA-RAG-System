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


def process_faq_file(file_path: str) -> List[tuple[Document, str]]:
    """
    Process a single FAQ JSON file and return a list of Documents and their IDs.
    Creates a separate document for each numbered section in the FAQ file.
    
    Args:
        file_path (str): Path to the FAQ JSON file
    
    Returns:
        List[tuple[Document, str]]: List of (Document, ID) tuples
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    file_name = os.path.basename(file_path)
    documents_and_ids = []
    
    for source_id, qa_list in data.items():
        # Collect all questions and create the content dictionary for this section
        section_questions = ''
        qa_dict = {}
        
        for qa_item in qa_list:
            question = qa_item.get('question', '').strip()
            answers = qa_item.get('answers', [])
            answer_text = ' '.join(answers).strip()
            
            if question and answer_text:
                section_questions = section_questions + " " + question
                qa_dict[question] = answer_text
        
        if qa_dict:
            
            # Create metadata
            metadata = {
                'category': 'faq',
                'file_source': file_name,
                'source_id': source_id,
                'questions': section_questions
            }
            
            # Create document
            doc = Document(
                page_content=json.dumps(qa_dict, ensure_ascii=False),
                metadata=metadata
            )
            
            documents_and_ids.append((doc, str(uuid4())))
    
    return documents_and_ids

def load_faq_to_chromadb(
    folder_path: str,
    vector_store,
    batch_size: int = 100
) -> Dict[str, int]:
    """
    Load all FAQ JSON files from a folder and save them to ChromaDB
    
    Args:
        folder_path (str): Path to the folder containing FAQ JSON files
        vector_store (Chroma): Initialized ChromaDB vector store
        batch_size (int): Number of documents to process at once
    
    Returns:
        Dict[str, int]: Dictionary with filenames and number of questions processed
    """
    all_documents = []
    all_ids = []
    files_processed = {}
    
    # Get all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return files_processed
    
    # Process each FAQ JSON file
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        try:
            docs_and_ids = process_faq_file(file_path)
            if docs_and_ids:
                # Process all documents from the file
                for doc, doc_id in docs_and_ids:
                    all_documents.append(doc)
                    all_ids.append(doc_id)
                
                # Sum up the total questions across all sections
                total_questions = sum(len(doc.metadata['questions']) for doc, _ in docs_and_ids)
                files_processed[json_file] = total_questions
            
        except Exception as e:
            print(f"Error processing FAQ file {json_file}: {str(e)}")
            continue
    
    # Save all documents to ChromaDB in batches
    total_batches = len(all_documents) // batch_size + (1 if len(all_documents) % batch_size != 0 else 0)
    
    for i in tqdm(range(0, len(all_documents), batch_size), desc="Saving FAQ to ChromaDB", total=total_batches):
        batch_documents = all_documents[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        
        vector_store.add_documents(
            documents=batch_documents,
            ids=batch_ids
        )
    
    return files_processed


def main():
    load_dotenv()
    client = QdrantClient(url=os.getenv("qdrant_url"), timeout=60)

    ### 創一個 collection
    # 如果不存在，創建 collection
    if not client.collection_exists("qa_dense_e5"):
        client.create_collection(
            collection_name="qa_dense_e5",
            vectors_config=VectorParams(
                size=1024, distance=Distance.COSINE
            ),  # intfloat/multilingual-e5-large 的參數設置 (FastEmbed 沒有所以自己設)
            on_disk_payload=True,
        )

    ### 做一個vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="qa_dense_e5",
        embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
        retrieval_mode=RetrievalMode.DENSE,
    )

    json_folder_path = "競賽資料集/reference/faq"
    load_faq_to_chromadb(json_folder_path, vector_store, batch_size=25)


if __name__ == "__main__":
    main()
