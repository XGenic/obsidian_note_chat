import os
import chromadb
from pathlib import Path

# --- PATH MANAGEMENT ---
class ChromaDBPathManager:
    def __init__(self, vault_root):
        self.vault_root = Path(vault_root).resolve()

    def get_canonical_path(self, file_path):
        raw_path = Path(file_path)
        path = (self.vault_root / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
        try:
            relative = path.relative_to(self.vault_root)
            return str(relative).replace('\\', '/')
        except ValueError:
            return str(path).replace('\\', '/')

# --- CONFIGURATION ---
OBSIDIAN_VAULT_PATH = "D:/Documents/Obsidian"
CHROMA_DB_PATH = "D:/Documents/chromadb"
COLLECTION_NAME = "obsidian_vault_main"

def verify_index_completeness():
    """
    Compares the files in the Obsidian vault with the indexed entries in ChromaDB
    to find discrepancies.
    """
    print("--- Starting Index Verification ---")
    path_manager = ChromaDBPathManager(OBSIDIAN_VAULT_PATH)

    # 1. Get all .md files from the Obsidian Vault, respecting ignore rules.
    print(f"Scanning vault path: {OBSIDIAN_VAULT_PATH}")
    disk_files = set()
    for root, dirs, files in os.walk(OBSIDIAN_VAULT_PATH):
        # Exclude directories starting with '!' or '.'
        dirs[:] = [d for d in dirs if not d.startswith(('!', '.'))]
        for filename in files:
            if filename.endswith(".md"):
                full_path = os.path.join(root, filename)
                disk_files.add(path_manager.get_canonical_path(full_path))
    print(f"Found {len(disk_files)} '.md' files in the vault.")

    # 2. Get all unique 'parent_file' metadata from ChromaDB.
    print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
    try:
        client_chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client_chroma.get_collection(name=COLLECTION_NAME)
        print(f"Connected to collection: '{COLLECTION_NAME}'")
        
        # Fetch all metadata from the collection
        # This could be memory intensive for very large collections, but is robust.
        all_records = collection.get(include=["metadatas"])
        
        indexed_files = set()
        for metadata in all_records['metadatas']:
            if 'parent_file' in metadata:
                indexed_files.add(path_manager.get_canonical_path(metadata['parent_file']))

        print(f"Found {len(indexed_files)} unique indexed files in ChromaDB.")

    except Exception as e:
        print(f"Error connecting to or querying ChromaDB: {e}")
        print("Verification cannot proceed.")
        return

    # 3. Compare the two sets to find discrepancies.
    print("\n--- Verification Results ---")
    
    unindexed_files = disk_files - indexed_files
    orphaned_entries = indexed_files - disk_files

    # --- Report Findings ---
    print(f"Total Files on Disk:      {len(disk_files)}")
    print(f"Total Files in Index:     {len(indexed_files)}")
    print("-" * 30)

    if not unindexed_files:
        print("[+] All files on disk are indexed.")
    else:
        print(f"[-] Found {len(unindexed_files)} Unindexed Files (exist on disk but not in DB):")
        for file_path in sorted(list(unindexed_files)):
            print(f"  - {file_path}")

    print("-" * 30)

    if not orphaned_entries:
        print("[+] No orphaned entries found in the index.")
    else:
        print(f"[-] Found {len(orphaned_entries)} Orphaned Entries (in DB but not on disk):")
        for file_path in sorted(list(orphaned_entries)):
            print(f"  - {file_path}")
            
    print("\n--- Verification Complete ---")


if __name__ == "__main__":
    verify_index_completeness()
