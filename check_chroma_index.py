import chromadb
import os
import hashlib
from pathlib import Path

# --- CONFIGURATION (from index_vault.py) ---
OBSIDIAN_VAULT_PATH = "D:/Documents/Obsidian"
CHROMA_DB_PATH = "D:/Documents/chromadb"
COLLECTION_NAME = "obsidian_vault_main"

# --- FILE TO CHECK ---
FILE_TO_CHECK = r"D:/Documents/Obsidian/Musings/Daily (not really)/2025/07/2025-07-21.md"


class ChromaDBPathManager:
    def __init__(self, vault_root):
        self.vault_root = Path(vault_root).resolve()

    def get_canonical_path(self, file_path):
        """Convert any path variant to the canonical format."""
        raw_path = Path(file_path)
        path = (self.vault_root / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()

        try:
            relative = path.relative_to(self.vault_root)
            canonical = str(relative).replace('\\', '/')
        except ValueError:
            canonical = str(path).replace('\\', '/')

        return canonical

    def get_path_id(self, file_path):
        canonical = self.get_canonical_path(file_path)
        return hashlib.md5(canonical.encode()).hexdigest()[:12]

def check_file_in_chroma():
    path_manager = ChromaDBPathManager(OBSIDIAN_VAULT_PATH)
    canonical_file_path = path_manager.get_canonical_path(FILE_TO_CHECK)
    absolute_file_path = str(Path(FILE_TO_CHECK).resolve()).replace('\\', '/')

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
    print(f"Canonical query path: {canonical_file_path}")

    canonical_results = collection.get(
        where={"parent_file": canonical_file_path},
        limit=1,
        include=[]
    )

    if canonical_results and canonical_results["ids"]:
        print(f"\nFile '{os.path.basename(FILE_TO_CHECK)}' IS indexed in ChromaDB.")
        print("Index format is current: parent_file metadata uses canonical vault-relative paths.")
        print(f"Found {len(canonical_results['ids'])} matching entries.")
    else:
        absolute_results = collection.get(
            where={"parent_file": absolute_file_path},
            limit=1,
            include=[]
        )

        if absolute_results and absolute_results["ids"]:
            print(f"\nFile '{os.path.basename(FILE_TO_CHECK)}' IS indexed in ChromaDB.")
            print("Index format is stale: parent_file metadata still uses absolute paths.")
            print("Run index_vault.py to rebuild the collection with canonical vault-relative paths.")
            print(f"Found {len(absolute_results['ids'])} matching entries.")
        else:
            print(f"\nFile '{os.path.basename(FILE_TO_CHECK)}' IS NOT indexed in ChromaDB.")
            print("If this file should exist in the index, run index_vault.py to rebuild it.")

if __name__ == "__main__":
    check_file_in_chroma()
