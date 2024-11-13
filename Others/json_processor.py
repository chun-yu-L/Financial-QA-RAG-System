import json
import os
from typing import List, Dict
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
                category = os.path.splitext(os.path.basename(file_path))[0].replace('corpus_', '')
                doc_id = f"{category}_{source_id}_page_{page_number}"
                
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
    
    for section_id, qa_list in data.items():
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
            # Create a unique ID for the section document
            doc_id = f"faq_{file_name}_section_{section_id}"
            
            # Create metadata
            metadata = {
                'category': 'faq',
                'file_source': file_name,
                'section_id': section_id,
                'questions': section_questions
            }
            
            # Create document
            doc = Document(
                page_content=json.dumps(qa_dict, ensure_ascii=False),
                metadata=metadata
            )
            
            documents_and_ids.append((doc, doc_id))
    
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

# Verification function
def verify_chromadb_content(vector_store) -> None:
    """
    Verify the content saved in ChromaDB for both FAQ and regular documents
    """
    try:
        # Get all documents
        collection_data = vector_store.get()
        total_documents = len(collection_data['ids'])
        
        # Separate FAQ and regular documents
        faq_docs = []
        regular_docs = []
        for i, metadata in enumerate(collection_data['metadatas']):
            if metadata.get('category') == 'faq':
                faq_docs.append(i)
            else:
                regular_docs.append(i)
        
        # Print summary
        print(f"Total documents in collection: {total_documents}")
        print(f"FAQ documents: {len(faq_docs)}")
        print(f"Regular documents: {len(regular_docs)}")
        
        # Show sample FAQ document
        if faq_docs:
            print("\n=== Sample FAQ Document ===")
            idx = faq_docs[0]
            metadata = collection_data['metadatas'][idx]
            print(f"ID: {collection_data['ids'][idx]}")
            print(f"File source: {metadata.get('file_source', 'N/A')}")
            print(f"Section ID: {metadata.get('section_id', 'N/A')}")
            print(f"Number of questions in section: {len(metadata.get('questions', []))}")
            
            # Parse and pretty print the content
            try:
                content = json.loads(collection_data['documents'][idx])
                print("Content (Q&A pairs in this section):")
                for q, a in content.items():
                    print(f"Q: {q}")
                    print(f"A: {a}\n")
            except json.JSONDecodeError:
                print("Error: Could not parse FAQ content as JSON")
        
        # ... (keep the rest of the verification function the same)
        
    except Exception as e:
        print(f"Error during verification: {e}")

# Usage example
if __name__ == "__main__":
    from Database import ChromaDB
    
    # Initialize ChromaDB
    chroma = ChromaDB()
    vector_store = chroma.get_vector_store(collection_name="test_collection")
    
    # Process regular JSON files
    json_folder_path = "./json_files"
    processed_files = load_json_folder_to_chromadb(json_folder_path, vector_store)
    
    # Process FAQ JSON files
    faq_folder_path = "./faq"
    processed_faq_files = load_faq_to_chromadb(faq_folder_path, vector_store)
    
    # Print summary
    print("\nProcessing Summary:")
    print("Regular JSON files:")
    total_docs = sum(processed_files.values())
    for filename, doc_count in processed_files.items():
        print(f"{filename}: {doc_count} documents")
    
    print("\nFAQ JSON files:")
    total_faq_docs = sum(processed_faq_files.values())
    for filename, qa_count in processed_faq_files.items():
        print(f"{filename}: {qa_count} Q&A pairs")
    
    print(f"\nTotal documents processed: {total_docs + total_faq_docs}")
    
    # Verify the saved content
    print("\nVerifying saved content:")
    verify_chromadb_content(vector_store)
