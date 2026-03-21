import os
from openai import OpenAI
import chromadb
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, Radiobutton, StringVar, Frame
from dotenv import load_dotenv
import re
import hashlib
from pathlib import Path
from date_utils import get_date_range_from_query

# --- CONFIGURATION ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CHROMA_DB_PATH = "D:/Documents/chromadb"
COLLECTION_NAME = "obsidian_vault_main"
RETRIEVAL_N_RESULTS = 30
KEYWORD_LIMIT = 20
FINAL_CONTEXT_DOCS = 12
last_focused_document = None
conversation_history = [] 
TITLE_MATCH_MIN_WORDS = 2
TITLE_MATCH_LIMIT = 3

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
api_clients = {}
if XAI_API_KEY:
    api_clients["Grok"] = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
if GEMINI_API_KEY:
    api_clients["Gemini"] = OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta")

# --- CHROMA DB EMBEDDING FUNCTION ---
class OpenAIEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key, model="text-embedding-3-small"):
        if not api_key:
            raise ValueError("OpenAI API key is missing.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, input_texts: list[str]) -> list[list[float]]:
        sanitized_input = [text if text.strip() else " " for text in input_texts]
        try:
            response = self.client.embeddings.create(input=sanitized_input, model=self.model)
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error calling OpenAI Embedding API: {e}")
            return [[] for _ in input_texts]

# --- SETUP ---
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
client_chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client_chroma.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
path_manager = ChromaDBPathManager("D:/Documents/Obsidian")

# --- DATE-BASED RETRIEVAL LOGIC ---

def get_all_parent_files():
    all_records = collection.get(include=["metadatas"])
    return {
        meta["parent_file"]
        for meta in all_records["metadatas"]
        if "parent_file" in meta
    }


def normalize_for_match(text):
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(normalized.split())


def is_date_style_title(title):
    return bool(re.fullmatch(r"\d{4} \d{2} \d{2}", title))


def find_title_matched_files(user_input):
    normalized_query = normalize_for_match(user_input)
    if not normalized_query:
        return []

    matched_files = []
    for parent_file in get_all_parent_files():
        base_name = Path(parent_file).stem
        normalized_title = normalize_for_match(base_name)
        if not normalized_title or is_date_style_title(normalized_title):
            continue

        word_count = len(normalized_title.split())
        if word_count < TITLE_MATCH_MIN_WORDS:
            continue

        if normalized_title in normalized_query:
            matched_files.append((word_count, len(normalized_title), parent_file))

    matched_files.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[2] for item in matched_files[:TITLE_MATCH_LIMIT]]

def handle_date_query(user_input):
    """
    Return (docs, metas) for daily notes whose filename falls into the
    detected date range.  If no dates are detected, or no daily notes
    match, return ([], []).
    """
    start_date, end_date = get_date_range_from_query(user_input)
    if start_date is None:
        return [], []                       # not a date query

    print("Date-aware query detected.")
    end_date = end_date.replace(hour=23, minute=59, second=59)
    print(f"Range: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")

    try:
        all_parent_files = get_all_parent_files()
    except Exception as e:
        print(f"ChromaDB error: {e}")
        return [], []

    date_pattern = re.compile(r".*?(\d{4}-\d{2}-\d{2})\.md$")
    target_files = set()

    for file_path in all_parent_files:
        m = date_pattern.match(file_path)
        if not m:
            continue
        try:
            file_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            if start_date.date() <= file_date <= end_date.date():
                target_files.add(file_path)
        except ValueError:
            continue

    if not target_files:
        print("No daily notes found in the specified range.")
        return [], []

    sorted_target_files = sorted(target_files)
    print(f"Matched daily notes: {sorted(os.path.basename(f) for f in sorted_target_files)}")

    ret_docs, ret_metas = [], []
    for fp in sorted_target_files:
        res = collection.get(where={"parent_file": {"$eq": fp}}, 
                             include=["documents", "metadatas"])
        print(f"[debug] querying parent_file={fp} -> {len(res['documents'])} docs")
        ret_docs.extend(res["documents"])
        ret_metas.extend(res["metadatas"])

    return ret_docs, ret_metas


is_processing = False

def send_message(event=None):
    global last_focused_document, conversation_history, is_processing
    
    if is_processing:
        return

    user_input = entry.get().strip()
    if not user_input:
        return
    
    if user_input.lower() in ["exit", "quit"]:
        root.quit()
        return

    try:
        is_processing = True
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, f"You: {user_input}\n", "user")
        entry.delete(0, tk.END)

        entry.config(state=tk.DISABLED)
        send_button.config(state=tk.DISABLED)

        # 1. Try date-aware retrieval
        final_documents, final_metadatas = handle_date_query(user_input)

        # 2. If no date docs, run hybrid semantic/keyword search
        if not final_documents:
            print("Date query returned nothing; running generic search.")
            query_text = "\n".join([turn['content'] for turn in conversation_history[-6:]]) + f"\n{user_input}"
            title_matched_files = find_title_matched_files(user_input)
            if title_matched_files:
                print(f"Title-matched files: {[os.path.basename(path) for path in title_matched_files]}")

            semantic_results = collection.query(
                query_texts=[query_text],
                n_results=RETRIEVAL_N_RESULTS,
                include=["metadatas", "documents"]
            )

            keyword_results = collection.get(
                where_document={"$contains": user_input},
                limit=KEYWORD_LIMIT,
                include=["metadatas", "documents"]
            )

            # de-duplicate
            final_docs = {}
            for parent_file in title_matched_files:
                title_results = collection.get(
                    where={"parent_file": {"$eq": parent_file}},
                    include=["metadatas", "documents"]
                )
                for i, doc_id in enumerate(title_results["ids"]):
                    if doc_id not in final_docs:
                        final_docs[doc_id] = {"doc": title_results["documents"][i],
                                              "meta": title_results["metadatas"][i]}
            for i, doc_id in enumerate(semantic_results["ids"][0]):
                if doc_id not in final_docs:
                    final_docs[doc_id] = {"doc": semantic_results["documents"][0][i],
                                          "meta": semantic_results["metadatas"][0][i]}
            for i, doc_id in enumerate(keyword_results["ids"]):
                if doc_id not in final_docs:
                    final_docs[doc_id] = {"doc": keyword_results["documents"][i],
                                          "meta": keyword_results["metadatas"][i]}

            initial_results = {
                "documents": [[item["doc"] for item in final_docs.values()]],
                "metadatas": [[item["meta"] for item in final_docs.values()]]
            }

            parent_files = [meta.get('parent_file')
                            for meta in initial_results["metadatas"][0]
                            if meta.get('parent_file')]

            # focus / broad-query logic (ONLY inside this else)
            if parent_files:
                broad_query_keywords = ["summary", "overall", "tl;dr", "tldr",
                                        "general opinion", "vibe", "arc",
                                        "the game", "the book", "the movie"]
                is_broad_query = any(k in user_input.lower() for k in broad_query_keywords)
                top_result_parent = parent_files[0]

                if is_broad_query:
                    print(f"Broad query detected. Focusing on top result: "
                          f"{os.path.basename(top_result_parent)}")
                    last_focused_document = top_result_parent
                    full_context_results = collection.get(
                        where={"parent_file": top_result_parent},
                        include=["documents", "metadatas"])
                    final_documents = full_context_results["documents"]
                    final_metadatas = full_context_results["metadatas"]
                else:
                    if top_result_parent:
                        last_focused_document = top_result_parent
                        print(f"Specific query. Focused document set to: "
                              f"{os.path.basename(last_focused_document)}")
                    else:
                        last_focused_document = None
                    final_documents = initial_results["documents"][0][:FINAL_CONTEXT_DOCS]
                    final_metadatas = initial_results["metadatas"][0][:FINAL_CONTEXT_DOCS]
            else:
                final_documents = []
                final_metadatas = []

        # 3. Build context string (works for both branches)
        print(f"Query: '{user_input}'")
        print("--- Retrieved Context ---")
        if final_documents:
            for i, meta in enumerate(final_metadatas):
                source_file = os.path.basename(meta.get('parent_file', 'Unknown File'))
                section_id = meta.get('section_id', 'N/A')
                source_type = meta.get('source_type', 'unknown')
                print(f"{i+1}. File: {source_file} (Type: {source_type}, Section: {section_id})")
        else:
            print("No relevant documents found in the vault.")

        context = ""
        for doc, meta in zip(final_documents, final_metadatas):
            source_file = os.path.basename(meta.get('parent_file', 'Unknown File'))
            section_info = f" (section: {meta.get('section_id')})" if meta.get('section_id') else ""
            context += f'''Context from my note '{source_file}'{section_info}:
{doc}

'''

        # 4. Call the LLM
        selected_model_name = model_choice.get()
        client = api_clients.get(selected_model_name)
        if not client:
            chat_history.insert(tk.END, f'Error: API client for {selected_model_name} not configured.\n\n', "error")
            chat_history.config(state=tk.DISABLED)
            return

        model_id = "grok-3-mini" if selected_model_name == "Grok" else "gemini-2.5-flash"
        today_str = datetime.now().strftime("%Y-%m-%d")

        system_prompt = (f'Today is {today_str}. Interpret relative time phrases against this date. '
                         'You are a helpful AI assistant, acting as a conversational partner '
                         'with a casual and loose tone. Your primary goal is to answer my question. '
                         'Use the context from my notes to enrich your answer if it is relevant; '
                         'do so naturally. Weave the information into the conversation rather than '
                         'just summarizing the notes. If the notes are not relevant, simply answer '
                         'my question directly using your general knowledge.')

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)

        user_prompt_content = f'''--- Context from my notes ---

{context}

--- My Question ---

{user_input}'''

        messages.append({"role": "user", "content": user_prompt_content})

        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=4096
            )
            reply = completion.choices[0].message.content

            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": reply})

            chat_history.insert(tk.END, f"{selected_model_name}: {reply}\n\n",
                                "grok" if selected_model_name == "Grok" else "gemini")
        except Exception as e:
            chat_history.insert(tk.END, f"Error: {e}\n\n", "error")

        chat_history.config(state=tk.DISABLED)
        chat_history.see(tk.END)

    finally:
        is_processing = False
        entry.config(state=tk.NORMAL)
        send_button.config(state=tk.NORMAL)
        entry.focus_set()

def save_transcript():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"C:\\Users\\Denis\\Desktop\\chat_transcript_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f'''Chat Transcript - {timestamp}\n\n''')
        f.write(chat_history.get("1.0", tk.END))
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f'''Saved to {filename}

''', "system")

# --- GUI SETUP ---
root = tk.Tk()
root.title("Multi-Model Vault Chat")
root.geometry("700x550")

model_frame = Frame(root)
model_frame.pack(pady=5)
model_choice = StringVar(value="Grok" if "Grok" in api_clients else "Gemini")

if "Grok" in api_clients:
    Radiobutton(model_frame, text="Grok", variable=model_choice, value="Grok").pack(side=tk.LEFT, padx=10)
if "Gemini" in api_clients:
    Radiobutton(model_frame, text="Gemini", variable=model_choice, value="Gemini").pack(side=tk.LEFT, padx=10)

chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=25, bg="#f0f0f0")
chat_history.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
chat_history.tag_config("user", foreground="blue")
chat_history.tag_config("grok", foreground="#006400")
chat_history.tag_config("gemini", foreground="#6a0dad")
chat_history.tag_config("error", foreground="red")
chat_history.tag_config("system", foreground="#555555")
chat_history.config(state=tk.DISABLED)

entry_frame = Frame(root)
entry_frame.pack(padx=10, pady=10, fill=tk.X)

entry = Entry(entry_frame, width=60)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
entry.bind("<Return>", send_message)

send_button = Button(entry_frame, text="Send", command=send_message)
send_button.pack(side=tk.LEFT, padx=5)

save_button = Button(entry_frame, text="Save Chat", command=save_transcript)
save_button.pack(side=tk.LEFT, padx=5)

root.mainloop()
