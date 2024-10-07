import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_core.documents import Document

def process_json_file(file_path: str) -> List[tuple[Document, str]]:
    """
    Process a single JSON file and return a list of Documents and their IDs
    
    Args:
        file_path (str): Path to the JSON file
    
    Returns:
        List[tuple[Document, str]]: List of (Document, ID) tuples
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    documents_and_ids = []
    
    for source_id, content in tqdm(data.items(), desc=f"Processing {os.path.basename(file_path)}"):
        metadata = content.get('metadata', {})
        pages = content.get('pages', [])
        
        for page in pages:
            page_number = page.get('page_number', 0)
            lines = page.get('lines', [])
            
            page_text = " ".join(lines)
            
            if page_text.strip():
                doc_id = f"{source_id}_page_{page_number}"
                
                page_metadata = {
                    **metadata,
                    'source_id': source_id,
                    'page_number': page_number,
                    'file_source': os.path.basename(file_path)
                }
                
                doc = Document(
                    page_content=page_text,
                    metadata=page_metadata
                )
                
                documents_and_ids.append((doc, doc_id))
    
    return documents_and_ids

def load_json_folder_to_chromadb(
    folder_path: str,
    vector_store,
    batch_size: int = 100
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
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return files_processed
    
    # Process each JSON file
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        try:
            docs_and_ids = process_json_file(file_path)
            documents, ids = zip(*docs_and_ids) if docs_and_ids else ([], [])
            
            all_documents.extend(documents)
            all_ids.extend(ids)
            
            files_processed[json_file] = len(documents)
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            continue
    
    # Save all documents to ChromaDB in batches
    total_batches = len(all_documents) // batch_size + (1 if len(all_documents) % batch_size != 0 else 0)
    
    for i in tqdm(range(0, len(all_documents), batch_size), desc="Saving to ChromaDB", total=total_batches):
        batch_documents = all_documents[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        
        vector_store.add_documents(
            documents=batch_documents,
            ids=batch_ids
        )
    
    return files_processed

def verify_chromadb_content(vector_store) -> None:
    """
    Verify the content saved in ChromaDB
    """
    try:
        # Get total count
        results = vector_store.similarity_search("", k=1)
        total_documents = len(vector_store.get()['ids'])
        print(f"Total documents in collection: {total_documents}")
        
        if total_documents > 0:
            # Show sample data
            print("\nSample document:")
            if results:
                sample_doc = results[0]
                print(f"Text snippet: {sample_doc.page_content[:200]}...")
                print(f"Metadata: {sample_doc.metadata}")
    except Exception as e:
        print(f"Error during verification: {e}")

# Usage example
if __name__ == "__main__":
    from Database import ChromaDB
    
    # Initialize ChromaDB
    chroma = ChromaDB()
    vector_store = chroma.get_vector_store(collection_name="test_collection")
    
    # Folder containing JSON files
    json_folder_path = "./json_files"  # Adjust this path to your folder
    
    # Load and save JSON data from all files
    processed_files = load_json_folder_to_chromadb(json_folder_path, vector_store)
    
    # Print summary
    print("\nProcessing Summary:")
    total_docs = sum(processed_files.values())
    for filename, doc_count in processed_files.items():
        print(f"{filename}: {doc_count} documents")
    print(f"Total documents processed: {total_docs}")
    
    # Verify the saved content
    print("\nVerifying saved content:")
    verify_chromadb_content(vector_store)