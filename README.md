# Obsidian Vault RAG Chat (Personalized)

This project is a highly customized, personal RAG (Retrieval-Augmented Generation) pipeline designed to facilitate a conversational interface with a specific Obsidian vault. It transforms a personal knowledge base into an intelligent, searchable companion that can answer questions and summarize information based on the user's own notes.

**Note:** This is not a general-purpose, plug-and-play tool. The indexing and retrieval logic is tightly coupled to the user's specific note-taking formats and vault structure. Adapting it for another vault would require significant modification of the classification and chunking rules in `index_vault.py`.

## Core Techniques & Implementation Details

This project employs several advanced techniques to provide relevant, context-aware responses.

*   **Dynamic, Multi-Tiered Indexing Strategy**: The system intelligently classifies notes and applies a different indexing strategy based on the note's structure and location:
    *   **Conversation Summaries**: For long conversation transcripts, it specifically extracts and indexes only the `## Overall Summary` section to capture the essence without noise.
    *   **Section-Based Chunking (for Structured Notes)**: For long, structured notes (e.g., book or project reviews), it uses a sophisticated chunking mechanism. It parses the document by headers (`## [[YYYY-MM-DD]]`, `## Episode X`), then uses `nltk` and `tiktoken` to split overly large sections and merge overly small ones, ensuring each chunk is semantically coherent and within an optimal token range.
    *   **Full Note Indexing**: For standard, shorter notes, the entire text is indexed as a single document.

*   **Stateful Conversational Memory**: The chat UI now maintains a complete, unabridged history of the current conversation. This entire history is sent with each new query, allowing the LLM to understand follow-up questions, pronouns, and the broader context of the dialogue. This leverages the large context windows of modern LLMs to create a truly stateful and natural conversational experience.

*   **Context-Aware Retrieval**: The database query itself is enhanced by the conversation history. It uses the last few turns of the dialogue to find documents that are relevant not just to the most recent question, but to the conversation as a whole.

*   **Two-Stage Dynamic Retrieval**: To handle both broad and specific queries, the chat interface uses a two-stage retrieval process:
    1.  An initial, broad query fetches the top 7 most relevant chunks from the database.
    2.  The system then analyzes these results. If a single parent document is "dominant" (i.e., constitutes >40% of the results) and the user's query is broad (e.g., "what's the summary of..."), it performs a second query to fetch *all* chunks related to that parent document.
    3.  Otherwise, it defaults to using the top 5 most relevant chunks from the initial search. This provides comprehensive context for summaries while maintaining precision for specific questions.

*   **Hierarchical Metadata**: All indexed chunks are stored in ChromaDB with rich metadata, including `parent_file` and `section_id`. This is critical for the two-stage retrieval logic, allowing the system to reassemble the full context of a document from its individual chunks.

*   **API-Powered & Multi-Model**:
    *   **Embeddings**: All text embedding is handled by the OpenAI `text-embedding-3-small` model via its API, removing the need for a powerful local GPU.
    *   **Generation**: The UI allows switching between different LLMs (currently Grok and Gemini) for generating responses.

*   **Incremental & Filtered Indexing**:
    *   The indexer checks the last modified time of files to only re-index new or changed notes, making subsequent runs fast and efficient.
    *   It can be configured to ignore specific folders (e.g., those prefixed with `!`), keeping the index clean and relevant.

## How It Works

1.  **`index_vault.py`**: This is the core indexing script. It walks through the Obsidian vault, classifies each note based on the rules described above, chunks it accordingly, generates embeddings via the OpenAI API, and `upserts` the data into a local ChromaDB vector database.

2.  **`chat_ui.py`**: This is the chat client. When a user sends a message, it embeds the query and uses the two-stage dynamic retrieval logic to find the most relevant context from ChromaDB. It then assembles a detailed prompt containing this context and sends it to the selected LLM (Grok or Gemini) to generate a response.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Install Dependencies

Ensure you have Python 3 installed. Then, install the required libraries from `requirements.txt`.

```bash
pip install -r requirements.txt
```

You may also need to download the `punkt` tokenizer for NLTK. This can be done by running the following in a Python interpreter:
```python
import nltk
nltk.download('punkt')
```

### 3. Configure API Keys

This project requires API keys from OpenAI (for embeddings) and at least one generation provider (Grok or Gemini).

1.  Create a file named `.env` in the project root directory.
2.  Add your API keys to the file:
    ```
    OPENAI_API_KEY="sk-YourActualOpenAIKey..."
    XAI_API_KEY="xai-YourActualXaiKey..."
    GEMINI_API_KEY="YourActualGeminiKey..."
    ```

### 4. Configure Paths

Open `index_vault.py` and `chat_ui.py` and adjust the following paths at the top of each file to match your system:

- `OBSIDIAN_VAULT_PATH`: The absolute path to your Obsidian vault.
- `CHROMA_DB_PATH`: The path where you want to store the vector database.

## Usage

1.  **Run the Indexer**: The first time you set up the project, you must index your entire vault. Run the indexer script from your terminal:

    ```bash
    python index_vault.py
    ```

    This may take some time depending on the size of your vault. Subsequent runs will be much faster.

2.  **Run the Chat UI**: Once indexing is complete, you can start the chat application:

    ```bash
    python chat_ui.py
    ```

    The GUI will appear, and you can start chatting with your notes!