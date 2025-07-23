

import chromadb
import os

# --- CONFIGURATION (from index_vault.py) ---
CHROMA_DB_PATH = "D:/Documents/chromadb"
COLLECTION_NAME = "obsidian_vault_main"

# --- FILE TO CHECK ---
FILE_TO_CHECK = r"C:/Users/Denis/Documents/Obsidian Vault\Musings\Daily (not really)\2025-07-21.md"

def check_file_in_chroma():
    print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
    client_chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        collection = client_chroma.get_collection(name=COLLECTION_NAME)
        print(f"Connected to collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Error connecting to collection '{COLLECTION_NAME}': {e}")
        print("Please ensure the collection exists and the path is correct.")
        return

    print(f"Checking for file: {FILE_TO_CHECK}")
    # Query for documents where the 'parent_file' metadata matches the file_path
    results = collection.get(
        where={"parent_file": FILE_TO_CHECK},
        limit=1, # We only need to know if at least one entry exists
        include=[] # No need to include embeddings or documents for this check
    )

    if results and results['ids']:
        print(f"\nFile '{os.path.basename(FILE_TO_CHECK)}' IS indexed in ChromaDB.")
        print(f"Found {len(results['ids'])} entries for this file.")
    else:
        print(f"\nFile '{os.path.basename(FILE_TO_CHECK)}' IS NOT indexed in ChromaDB.")

if __name__ == "__main__":
    check_file_in_chroma()

