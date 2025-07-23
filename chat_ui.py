import os
from openai import OpenAI
import chromadb
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, Radiobutton, StringVar, Frame
from dotenv import load_dotenv
from collections import Counter
from dateparser.search import search_dates
import re

# --- CONFIGURATION ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CHROMA_DB_PATH = "D:/Documents/chromadb"
COLLECTION_NAME = "obsidian_vault_main"
last_focused_document = None
conversation_history = [] 

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

# --- DATE-BASED RETRIEVAL LOGIC ---
def get_date_range_from_query(user_input):
    today = datetime.now()
    user_input_lower = user_input.lower()

    if "last week" in user_input_lower or "past week" in user_input_lower:
        start_of_last_week = today - timedelta(days=today.weekday() + 7)
        end_of_last_week = start_of_last_week + timedelta(days=6)
        return start_of_last_week, end_of_last_week
    
    if "yesterday" in user_input_lower:
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday

    parsed_dates = search_dates(
        user_input, 
        settings={'PREFER_DATES_FROM': 'past', 'RETURN_AS_TIMEZONE_AWARE': False}
    )

    if parsed_dates:
        if len(parsed_dates) > 1:
            start_date = min(d[1] for d in parsed_dates)
            end_date = max(d[1] for d in parsed_dates)
            return start_date, end_date
        else:
            return parsed_dates[0][1], parsed_dates[0][1]

    return None, None

def handle_date_query(user_input):
    start_date, end_date = get_date_range_from_query(user_input)

    if not start_date:
        return None, None

    print("Date-based query detected.")
    end_date = end_date.replace(hour=23, minute=59, second=59)
    print(f"Date Range Identified: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    try:
        all_records = collection.get(include=["metadatas"])
        all_parent_files = {
            meta['parent_file'].replace('\\', '/') 
            for meta in all_records['metadatas'] 
            if 'parent_file' in meta
        }
    except Exception as e:
        print(f"Error fetching all records from ChromaDB: {e}")
        return None, None

    target_files = set()
    date_pattern = re.compile(r'.*?(\\d{4}-\\d{2}-\\d{2})\\.md$')
    
    for file_path in all_parent_files:
        match = date_pattern.match(file_path)
        if match:
            try:
                file_date_str = match.group(1)
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                if start_date.date() <= file_date.date() <= end_date.date():
                    target_files.add(file_path)
            except ValueError:
                continue

    if not target_files:
        print("No daily notes found in the specified date range.")
        return [], []

    print(f"Found {len(target_files)} matching daily notes:")
    for f in sorted(list(target_files)):
        print(f"  - {os.path.basename(f)}")

    retrieved_docs = []
    retrieved_metas = []
    for file_path in target_files:
        results = collection.get(where={"parent_file": file_path}, include=["documents", "metadatas"])
        retrieved_docs.extend(results['documents'])
        retrieved_metas.extend(results['metadatas'])

    return retrieved_docs, retrieved_metas


# --- CHAT LOGIC ---
def send_message(event=None):
    global last_focused_document, conversation_history
    user_input = entry.get().strip()
    if not user_input: return
    if user_input.lower() in ["exit", "quit"]: root.quit(); return
    
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {user_input}\n", "user")
    entry.delete(0, tk.END)

    final_documents, final_metadatas = handle_date_query(user_input)

    if final_documents is None:
        query_text = "\n".join([turn['content'] for turn in conversation_history[-6:]]) + f"\n{user_input}"

        semantic_results = collection.query(
            query_texts=[query_text],
            n_results=15,
            include=["metadatas", "documents"]
        )

        keyword_results = collection.get(
            where_document={"$contains": user_input},
            limit=10,
            include=["metadatas", "documents"]
        )

        final_docs = {}
        for i, doc_id in enumerate(semantic_results["ids"][0]):
            if doc_id not in final_docs:
                final_docs[doc_id] = {"doc": semantic_results["documents"][0][i], "meta": semantic_results["metadatas"][0][i]}
        for i, doc_id in enumerate(keyword_results["ids"]):
            if doc_id not in final_docs:
                final_docs[doc_id] = {"doc": keyword_results["documents"][i], "meta": keyword_results["metadatas"][i]}

        initial_results = {
            "documents": [[item["doc"] for item in final_docs.values()]],
            "metadatas": [[item["meta"] for item in final_docs.values()]]
        }

        parent_files = [meta.get('parent_file') for meta in initial_results["metadatas"][0] if meta.get('parent_file')]
        
        final_documents = initial_results["documents"][0]
        final_metadatas = initial_results["metadatas"][0]

        if parent_files:
            broad_query_keywords = ["summary", "overall", "tl;dr", "tldr", "general opinion", "vibe", "arc", "the game", "the book", "the movie"]
            is_broad_query = any(keyword in user_input.lower() for keyword in broad_query_keywords)
            top_result_parent = parent_files[0] if parent_files else None

            if is_broad_query and last_focused_document:
                print(f"Broad query detected. Reusing last focused document: {os.path.basename(last_focused_document)}")
                full_context_results = collection.get(where={"parent_file": last_focused_document}, include=["documents", "metadatas"])
                final_documents = full_context_results["documents"]
                final_metadatas = full_context_results["metadatas"]
            
            elif is_broad_query and top_result_parent:
                print(f"Broad query with no focus. Focusing on top result: {os.path.basename(top_result_parent)}")
                last_focused_document = top_result_parent
                full_context_results = collection.get(where={"parent_file": top_result_parent}, include=["documents", "metadatas"])
                final_documents = full_context_results["documents"]
                final_metadatas = full_context_results["metadatas"]

            else:
                if top_result_parent:
                    last_focused_document = top_result_parent
                    print(f"Specific query. Focused document set to: {os.path.basename(last_focused_document)}")
                else:
                    last_focused_document = None
                final_documents = initial_results["documents"][0][:8]
                final_metadatas = initial_results["metadatas"][0][:8]

    print(f"Query: '{user_input}'")
    print("--- Retrieved Context ---")
    if not final_documents:
        print("No relevant documents found in the vault.")
        context = ""
    else:
        for i, meta in enumerate(final_metadatas):
            source_file = os.path.basename(meta.get('parent_file', 'Unknown File'))
            section_id = meta.get('section_id', 'N/A')
            source_type = meta.get('source_type', 'unknown')
            print(f"{i+1}. File: {source_file} (Type: {source_type}, Section: {section_id})")
        print("-" * 25)
        
        context = ""
        for doc, meta in zip(final_documents, final_metadatas):
            source_file = os.path.basename(meta.get('parent_file', 'Unknown File'))
            section_info = f" (section: {meta.get('section_id')})" if meta.get('section_id') else ""
            context += f'''Context from my note '{source_file}'{section_info}:
{doc}

'''

    selected_model_name = model_choice.get()
    client = api_clients.get(selected_model_name)
    
    if not client:
        chat_history.insert(tk.END, f'''Error: API client for {selected_model_name} not configured.\n\n''', "error")
        chat_history.config(state=tk.DISABLED)
        return

    model_id = "grok-3-mini" if selected_model_name == "Grok" else "models/gemini-1.5-flash-latest" 
    
    system_prompt = f'''You are a helpful AI assistant, acting as a conversational parter with a casual and loose tone. Your primary goal is to answer my question. Use the context from my notes to enrich your answer if it's relevant, do so naturally. Weave the information into the conversation rather than just summarizing the notes. If the notes are not relevant, simply answer my question directly using your general knowledge.'''
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    
    user_prompt_content = f'''--- Context from my notes ---

{context}

--- My Question ---

{user_input}
'''
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

        chat_history.insert(tk.END, f'''{selected_model_name}: {reply}\n\n''', "grok" if selected_model_name == "Grok" else "gemini")
    except Exception as e:
        chat_history.insert(tk.END, f'''Error: {e}\n\n''', "error")
    
    chat_history.config(state=tk.DISABLED)
    chat_history.see(tk.END)

def save_transcript():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"C:\\Users\\Denis\\Desktop\\chat_transcript_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f'''Chat Transcript - {timestamp}\n\n''')
        f.write(chat_history.get("1.0", tk.END))
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f'''Saved to {filename}\n\n''', "system")

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