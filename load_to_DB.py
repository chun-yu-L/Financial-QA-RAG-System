from Database import ChromaDB
from json_processor import load_json_folder_to_chromadb, verify_chromadb_content

# Initialize ChromaDB
chroma = ChromaDB()
vector_store = chroma.get_vector_store(collection_name="test_collection")

# Specify the folder containing JSON files
json_folder_path = "D:/2024_AI_cup/競賽資料集/reference/"  # Adjust this path to your folder

# Process all JSON files in the folder
processed_files = load_json_folder_to_chromadb(json_folder_path, vector_store)

# Verify the saved content
verify_chromadb_content(vector_store)