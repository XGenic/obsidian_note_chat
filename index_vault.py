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

# --- API CLIENTS ---
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta")

# --- LLM-POWERED TAGGING ---
def generate_tags_with_llm(text_content, client):
    if not client:
        print("  -> LLM client for tagging not configured. Skipping tag generation.")
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
            model="models/gemini-1.5-flash-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.2,
        )
        raw_tags = completion.choices[0].message.content.strip()
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
def classify_note(content, filename):
    """
    Classifies a note based on a set of hardcoded rules.
    If no rules match, it defaults to 'generic_note'.
    """
    # Rule-based classification
    if "## Overall Summary" in content and "# Transcript" in content:
        return "conversation"
    if re.search(r'##\s*(?:\b\d{4}-\d{2}-\d{2}\b|Episode\s*\d+|Chapter\s*\d+)', content):
        return "review_journal"
    if re.match(r'\d{4}-\d{2}-\d{2}\.md', filename) and "# Thoughts Through The Day" in content:
        return "daily_note"
        
    # Fallback to generic_note if no other rules match
    return "generic_note"


# --- CHUNKING LOGIC ---
def chunk_text_by_sections(text, min_tokens=100, max_tokens=1000):
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
            i += 1
            current_text += "\n" + chunks[i]
            current_meta["section_id"] += f"; {metadatas[i]['section_id']}"
        final_chunks.append(current_text)
        final_metadatas.append(current_meta)
        i += 1
        
    return final_chunks, final_metadatas

# --- MAIN SCRIPT LOGIC ---
def main():
    print("--- Starting Obsidian Vault Indexing ---")

    path_manager = ChromaDBPathManager(OBSIDIAN_VAULT_PATH)
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
    client_chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    print("Clearing old collection to ensure clean re-indexing...")
    client_chroma.delete_collection(name=COLLECTION_NAME)
    collection = client_chroma.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    for root, dirs, files in os.walk(OBSIDIAN_VAULT_PATH):
        dirs[:] = [d for d in dirs if not d.startswith(('!','.'))]
        for filename in files:
            if not filename.endswith(".md"): continue

            file_path = os.path.join(root, filename)
            canonical_path = path_manager.get_canonical_path(file_path)
            path_id = path_manager.get_path_id(file_path)
            
            print(f"[PROCESSING] '{canonical_path}'...")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            current_mtime = os.path.getmtime(file_path)
            note_type = classify_note(content, os.path.basename(filename))
            print(f"  -> Classified as: {note_type}")

            base_metadata = {
                "mtime": current_mtime,
                "parent_file": canonical_path,
                "parent_file_id": path_id
            }

            if note_type == "conversation":
                try:
                    summary_marker = "## Overall Summary"
                    transcript_marker = "# Transcript"
                    summary_start = content.find(summary_marker)
                    transcript_start = content.find(transcript_marker, summary_start)
                    
                    if summary_start != -1:
                        summary_content = content[summary_start + len(summary_marker):transcript_start].strip()
                        tags = generate_tags_with_llm(summary_content, gemini_client)
                        
                        metadata = base_metadata.copy()
                        metadata.update({
                            "source_type": "conversation_summary",
                            "tags": ", ".join(tags)
                        })

                        collection.upsert(
                            documents=[summary_content],
                            ids=[f"{path_id}_summary"],
                            metadatas=[metadata]
                        )
                        print(f"  -> Indexed summary for '{canonical_path}'.")
                    else:
                         print(f"  -> Warning: Summary marker not found in '{canonical_path}'. Skipping.")
                except Exception as e:
                    print(f"  -> Error parsing summary for '{canonical_path}': {e}. Skipping.")

            elif note_type in ["review_journal", "daily_note"]:
                chunks, section_metadatas = chunk_text_by_sections(content)
                if chunks:
                    chunk_ids = [f"{path_id}_chunk{i+1}" for i in range(len(chunks))]
                    chunk_metadatas = []
                    for i, chunk in enumerate(chunks):
                        tags = generate_tags_with_llm(chunk, gemini_client)
                        meta = base_metadata.copy()
                        meta.update({
                            "source_type": f"{note_type}_section",
                            "section_id": section_metadatas[i]["section_id"],
                            "tags": ", ".join(tags)
                        })
                        chunk_metadatas.append(meta)
                    
                    collection.upsert(documents=chunks, ids=chunk_ids, metadatas=chunk_metadatas)
                    print(f"  -> Indexed {len(chunks)} chunks for '{canonical_path}'.")
                else:
                    print(f"  -> No sections found to chunk in '{canonical_path}'. Skipping.")

            else: # generic_note
                tags = generate_tags_with_llm(content, gemini_client)
                metadata = base_metadata.copy()
                metadata.update({
                    "source_type": "full_note",
                    "tags": ", ".join(tags)
                })
                collection.upsert(
                    documents=[content],
                    ids=[path_id],
                    metadatas=[metadata]
                )
                print(f"  -> Indexed full note for '{canonical_path}'.")

    print("--- Indexing Complete ---")

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
    main()
