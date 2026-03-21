import argparse
import os
import re
import chromadb
import tiktoken
import nltk
import numpy as np
from openai import OpenAI
import time
from dotenv import load_dotenv
import json
import hashlib
from pathlib import Path

# --- CONFIGURATION ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OBSIDIAN_VAULT_PATH = "D:/Documents/Obsidian"
CHROMA_DB_PATH = "D:/Documents/chromadb"
COLLECTION_NAME = "obsidian_vault_main"

# --- CHUNKING TUNING ---
CHUNK_MIN_TOKENS = 250
CHUNK_MAX_TOKENS = 2200
GENERIC_NOTE_CHUNK_MAX_TOKENS = 2200

# --- PATH MANAGEMENT ---
class ChromaDBPathManager:
    def __init__(self, vault_root):
        self.vault_root = Path(vault_root).resolve()
    
    def get_canonical_path(self, file_path):
        """Convert any path variant to the canonical format."""
        raw_path = Path(file_path)
        path = (self.vault_root / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
        
        try:
            # Try to make it relative to vault root
            relative = path.relative_to(self.vault_root)
            canonical = str(relative).replace('\\', '/')
        except ValueError:
            # File outside vault, use absolute path
            canonical = str(path).replace('\\', '/')
        
        return canonical
    
    def get_path_id(self, file_path):
        """Generate a unique ID for any path variant."""
        canonical = self.get_canonical_path(file_path)
        return hashlib.md5(canonical.encode()).hexdigest()[:12]


def get_file_signature(file_path):
    """Return a stable file signature for incremental indexing checks."""
    stat_result = os.stat(file_path)
    return {
        "mtime": stat_result.st_mtime,
        "mtime_ns": int(stat_result.st_mtime_ns),
        "size_bytes": int(stat_result.st_size),
    }


def compute_content_hash(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_indexed_file_state(existing_metadatas):
    indexed_files = {}
    indexed_parent_files = set()

    for metadata in existing_metadatas:
        parent_file = metadata.get("parent_file")
        parent_file_id = metadata.get("parent_file_id")
        if not parent_file or not parent_file_id:
            continue

        indexed_parent_files.add(parent_file)
        state = indexed_files.setdefault(parent_file_id, {})
        state["parent_file"] = parent_file
        state["mtime"] = metadata.get("mtime")
        state["mtime_ns"] = metadata.get("mtime_ns")
        state["size_bytes"] = metadata.get("size_bytes")
        state["content_hash"] = metadata.get("content_hash")

    return indexed_files, indexed_parent_files


def should_skip_file(existing_state, current_signature):
    if not existing_state:
        return False, "new file"

    stored_mtime_ns = existing_state.get("mtime_ns")
    stored_size = existing_state.get("size_bytes")
    if stored_mtime_ns is not None and stored_size is not None:
        if int(stored_mtime_ns) == current_signature["mtime_ns"] and int(stored_size) == current_signature["size_bytes"]:
            return True, "mtime_ns + size match"

    stored_mtime = existing_state.get("mtime")
    if stored_mtime is not None and stored_size is not None:
        if stored_mtime == current_signature["mtime"] and int(stored_size) == current_signature["size_bytes"]:
            return True, "legacy mtime + size match"

    return False, "file signature changed"

# --- API CLIENTS ---
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta")

# --- LLM-POWERED TAGGING ---
def generate_tags_with_llm(text_content, client):
    if not client:
        print("  -> LLM client for tagging not configured. Skipping tag generation.")
        return []

    if not text_content or not text_content.strip():
        print("  -> Text is empty. Skipping tag generation.")
        return []
    
    prompt = f"""Analyze the following text and generate 3-5 relevant keyword tags that capture the main topics, themes, and entities. Return the tags as a single comma-separated string.

For example: "tag1, tag2, tag3, tag4"

Text:
---
{text_content[:4000]}
---

Tags:"""
    
    try:
        print("  -> Generating tags with LLM...")
        completion = client.chat.completions.create(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.2,
        )
        choice = completion.choices[0] if completion.choices else None
        if not choice or not choice.message or not choice.message.content:
            finish_reason = getattr(choice, "finish_reason", "unknown")
            print(f"  -> Tag generation returned no content (finish_reason={finish_reason}). Skipping.")
            return []

        raw_tags = choice.message.content.strip()
        # Clean up the tags and return as a list of strings
        tags = [tag.strip() for tag in raw_tags.split(',') if tag.strip()]
        print(f"  -> Generated Tags: {tags}")
        return tags
    except Exception as e:
        print(f"  -> Error generating tags: {e}. Skipping.")
        return []

# --- CHROMA DB EMBEDDING FUNCTION ---
class OpenAIEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key, model="text-embedding-3-small"):
        if not api_key:
            raise ValueError("OpenAI API key is missing.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_dim = 1536

    def __call__(self, input_texts: list[str]) -> list[list[float]]:
        sanitized_input = [text if text.strip() else " " for text in input_texts]
        try:
            response = self.client.embeddings.create(input=sanitized_input, model=self.model)
            if not response.data or not response.data[0].embedding:
                raise ValueError("API returned no embeddings")
            
            embedding_map = {item.index: item.embedding for item in response.data}
            return [embedding_map.get(i, [0.0] * self.embedding_dim) for i in range(len(sanitized_input))]
        except Exception as e:
            print(f"Error calling OpenAI Embedding API: {e}. Returning zero-vectors.")
            return [[0.0] * self.embedding_dim for _ in sanitized_input]

# --- NOTE CLASSIFICATION LOGIC ---
def extract_conversation_summary(content):
    """Extract the text between the summary and transcript headings, case-insensitively."""
    summary_match = re.search(r'(?im)^##\s*overall summary\s*$', content)
    transcript_match = re.search(r'(?im)^#\s*transcript\s*$', content)

    if not summary_match or not transcript_match or transcript_match.start() <= summary_match.end():
        return None

    return content[summary_match.end():transcript_match.start()].strip()


def classify_note(content, filename):
    """
    Classifies a note based on a set of hardcoded rules.
    If no rules match, it defaults to 'generic_note'.
    """
    # Rule-based classification
    if extract_conversation_summary(content) is not None:
        return "conversation"
    if re.search(r'##\s*(?:\b\d{4}-\d{2}-\d{2}\b|Episode\s*\d+|Chapter\s*\d+)', content):
        return "review_journal"
    if re.match(r'\d{4}-\d{2}-\d{2}\.md', filename) and "# Thoughts Through The Day" in content:
        return "daily_note"
        
    # Fallback to generic_note if no other rules match
    return "generic_note"


# --- CHUNKING LOGIC ---
def chunk_text_by_sections(text, min_tokens=CHUNK_MIN_TOKENS, max_tokens=CHUNK_MAX_TOKENS):
    encoder = tiktoken.get_encoding("cl100k_base")
    section_pattern = r'(##\s*(?:\b\d{4}-\d{2}-\d{2}\b|Episode\s*\d+|Chapter\s*\d+).*?)(?=(##\s*(?:\b\d{4}-\d{2}-\d{2}\b|Episode\s*\d+|Chapter\s*\d+)|$))'
    sections = re.findall(section_pattern, text, re.DOTALL)
    
    chunks, metadatas = [], []
    if not sections:
        return [text], [{"section_id": "full_document"}]

    for section_content, _ in sections:
        section_header_match = re.match(r'##\s*(.*)', section_content)
        section_header = section_header_match.group(1).strip() if section_header_match else "Unnamed Section"
        
        if len(encoder.encode(section_content)) > max_tokens:
            sentences = nltk.sent_tokenize(section_content)
            temp_chunk, temp_tokens = [], 0
            for i, sentence in enumerate(sentences):
                sentence_tokens = len(encoder.encode(sentence))
                if temp_tokens + sentence_tokens <= max_tokens:
                    temp_chunk.append(sentence)
                    temp_tokens += sentence_tokens
                else:
                    chunks.append(" ".join(temp_chunk))
                    metadatas.append({"section_id": f"{section_header} (part {i})"})
                    temp_chunk, temp_tokens = [sentence], sentence_tokens
            if temp_chunk:
                chunks.append(" ".join(temp_chunk))
                metadatas.append({"section_id": f"{section_header} (part {len(sentences)})"})
        else:
            chunks.append(section_content)
            metadatas.append({"section_id": section_header})

    final_chunks, final_metadatas = [], []
    i = 0
    while i < len(chunks):
        current_text, current_meta = chunks[i], metadatas[i]
        while len(encoder.encode(current_text)) < min_tokens and i + 1 < len(chunks):
            candidate_text = current_text + "\n" + chunks[i + 1]
            if len(encoder.encode(candidate_text)) > max_tokens:
                break
            i += 1
            current_text = candidate_text
            current_meta["section_id"] += f"; {metadatas[i]['section_id']}"
        final_chunks.append(current_text)
        final_metadatas.append(current_meta)
        i += 1
        
    return final_chunks, final_metadatas


def chunk_text_fixed_chunks(text, max_tokens=GENERIC_NOTE_CHUNK_MAX_TOKENS, min_tokens=100):
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    text_tokens = encoder.encode(text)
    start = 0
    while start < len(text_tokens):
        end = min(start + max_tokens, len(text_tokens))
        chunk_tokens = text_tokens[start:end]
        chunks.append(encoder.decode(chunk_tokens))
        start = end

    # merge tiny tail chunk
    if len(chunks) > 1 and len(encoder.encode(chunks[-1])) < min_tokens:
        chunks[-2] += "\n" + chunks[-1]
        chunks.pop()

    return chunks


def build_note_records(content, canonical_path, path_id, file_signature, content_hash):
    note_type = classify_note(content, Path(canonical_path).name)
    print(f"  -> Classified as: {note_type}")

    base_metadata = {
        "mtime": file_signature["mtime"],
        "mtime_ns": file_signature["mtime_ns"],
        "size_bytes": file_signature["size_bytes"],
        "content_hash": content_hash,
        "parent_file": canonical_path,
        "parent_file_id": path_id
    }

    documents = []
    ids = []
    metadatas = []

    if note_type == "conversation":
        summary_content = extract_conversation_summary(content)
        if summary_content is None:
            print(f"  -> Warning: Conversation summary/transcript headings not found in '{canonical_path}'. Skipping.")
            return [], [], []

        tags = generate_tags_with_llm(summary_content, gemini_client)
        metadata = base_metadata.copy()
        metadata.update({
            "source_type": "conversation_summary",
            "tags": ", ".join(tags)
        })

        documents.append(summary_content)
        ids.append(f"{path_id}_summary")
        metadatas.append(metadata)
        print(f"  -> Prepared summary for '{canonical_path}'.")
        return documents, ids, metadatas

    if note_type in ["review_journal", "daily_note"]:
        chunks, section_metadatas = chunk_text_by_sections(content)
        if not chunks:
            print(f"  -> No sections found to chunk in '{canonical_path}'. Skipping.")
            return [], [], []

        for i, chunk in enumerate(chunks):
            tags = generate_tags_with_llm(chunk, gemini_client)
            metadata = base_metadata.copy()
            metadata.update({
                "source_type": f"{note_type}_section",
                "section_id": section_metadatas[i]["section_id"],
                "tags": ", ".join(tags)
            })
            documents.append(chunk)
            ids.append(f"{path_id}_chunk{i+1}")
            metadatas.append(metadata)

        print(f"  -> Prepared {len(chunks)} chunks for '{canonical_path}'.")
        return documents, ids, metadatas

    chunks = chunk_text_fixed_chunks(content)
    for i, chunk in enumerate(chunks):
        tags = generate_tags_with_llm(chunk, gemini_client)
        metadata = base_metadata.copy()
        metadata.update({
            "source_type": "generic_note_chunk",
            "section_id": f"chunk_{i+1}",
            "tags": ", ".join(tags)
        })
        documents.append(chunk)
        ids.append(f"{path_id}_chunk{i+1}")
        metadatas.append(metadata)

    print(f"  -> Prepared {len(chunks)} chunks for generic note '{canonical_path}'.")
    return documents, ids, metadatas


def get_or_create_collection(client_chroma, embedding_function):
    try:
        print("Loading existing collection for incremental indexing...")
        return client_chroma.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
    except Exception:
        print("Collection not found. Creating a new one...")
        return client_chroma.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )


def reset_collection(client_chroma, embedding_function):
    print("Full reindex requested. Rebuilding collection from scratch...")
    try:
        client_chroma.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    return client_chroma.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

# --- MAIN SCRIPT LOGIC ---
def main(full_reindex=False):
    print("--- Starting Obsidian Vault Indexing ---")

    path_manager = ChromaDBPathManager(OBSIDIAN_VAULT_PATH)
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
    client_chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection = (
        reset_collection(client_chroma, embedding_function)
        if full_reindex
        else get_or_create_collection(client_chroma, embedding_function)
    )

    existing_records = collection.get(include=["metadatas"])
    existing_metadatas = existing_records.get("metadatas", [])
    indexed_files, indexed_parent_files = build_indexed_file_state(existing_metadatas)

    seen_parent_files = set()
    processed_count = 0
    skipped_count = 0

    for root, dirs, files in os.walk(OBSIDIAN_VAULT_PATH):
        dirs[:] = [d for d in dirs if not d.startswith(('!','.'))]
        for filename in files:
            if not filename.endswith(".md"): continue

            file_path = os.path.join(root, filename)
            canonical_path = path_manager.get_canonical_path(file_path)
            path_id = path_manager.get_path_id(file_path)
            seen_parent_files.add(canonical_path)
            
            print(f"[PROCESSING] '{canonical_path}'...")

            current_signature = get_file_signature(file_path)
            existing_state = indexed_files.get(path_id)
            should_skip, skip_reason = should_skip_file(existing_state, current_signature)
            if should_skip:
                print(f"  -> Unchanged ({skip_reason}). Skipping.")
                skipped_count += 1
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            current_content_hash = compute_content_hash(content)
            stored_content_hash = existing_state.get("content_hash") if existing_state else None
            if stored_content_hash and stored_content_hash == current_content_hash:
                print(
                    "  -> Content unchanged; refreshing metadata only "
                    f"(stored mtime_ns={existing_state.get('mtime_ns')}, current mtime_ns={current_signature['mtime_ns']})."
                )
                existing_note_records = collection.get(
                    where={"parent_file_id": {"$eq": path_id}},
                    include=["metadatas"]
                )
                existing_ids = existing_note_records.get("ids", [])
                updated_metadatas = []
                for metadata in existing_note_records.get("metadatas", []):
                    refreshed_metadata = dict(metadata)
                    refreshed_metadata.update({
                        "mtime": current_signature["mtime"],
                        "mtime_ns": current_signature["mtime_ns"],
                        "size_bytes": current_signature["size_bytes"],
                        "content_hash": current_content_hash,
                    })
                    updated_metadatas.append(refreshed_metadata)

                if existing_ids and updated_metadatas:
                    collection.update(ids=existing_ids, metadatas=updated_metadatas)
                    skipped_count += 1
                    print("  -> Updated stored file signature without re-embedding.")
                    continue

            if existing_state:
                print(
                    "  -> Reindexing due to signature change "
                    f"(stored mtime_ns={existing_state.get('mtime_ns')}, current mtime_ns={current_signature['mtime_ns']}, "
                    f"stored size={existing_state.get('size_bytes')}, current size={current_signature['size_bytes']})."
                )

            existing_note_records = collection.get(
                where={"parent_file_id": {"$eq": path_id}},
                include=["metadatas"]
            )
            existing_ids = existing_note_records.get("ids", [])
            if existing_ids:
                collection.delete(ids=existing_ids)
                print(f"  -> Removed {len(existing_ids)} stale indexed chunks.")

            try:
                documents, ids, metadatas = build_note_records(
                    content,
                    canonical_path,
                    path_id,
                    current_signature,
                    current_content_hash
                )
                if ids:
                    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
                    processed_count += 1
                    print(f"  -> Upserted {len(ids)} records.")
                else:
                    print(f"  -> No indexable content produced for '{canonical_path}'.")
            except Exception as e:
                print(f"  -> Error indexing '{canonical_path}': {e}. Skipping.")

    orphaned_parent_files = indexed_parent_files - seen_parent_files
    for orphaned_parent_file in sorted(orphaned_parent_files):
        orphaned_path_id = path_manager.get_path_id(orphaned_parent_file)
        orphaned_records = collection.get(
            where={"parent_file_id": {"$eq": orphaned_path_id}},
            include=["metadatas"]
        )
        orphaned_ids = orphaned_records.get("ids", [])
        if orphaned_ids:
            collection.delete(ids=orphaned_ids)
            print(f"[REMOVED] Deleted {len(orphaned_ids)} orphaned records for '{orphaned_parent_file}'.")

    print(f"Processed changed files: {processed_count}")
    print(f"Skipped unchanged files: {skipped_count}")
    print(f"Removed orphaned files: {len(orphaned_parent_files)}")
    print("--- Indexing Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index an Obsidian vault into ChromaDB.")
    parser.add_argument(
        "--full-reindex",
        action="store_true",
        help="Delete and rebuild the entire Chroma collection instead of updating incrementally."
    )
    args = parser.parse_args()

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
    main(full_reindex=args.full_reindex)
