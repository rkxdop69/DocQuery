# DocQuery

A full-stack **Retrieval-Augmented Generation (RAG)** web app that lets you upload PDF or TXT documents and ask natural language questions about them. Built with Flask, FAISS, sentence-transformers, and the Gemini API.

---

## Features

- **Document ingestion** — Upload PDF or TXT files; text is extracted, chunked, embedded, and stored in a FAISS vector index
- **Semantic search** — Questions are embedded and matched against document chunks using L2 similarity
- **Source tracking** — Every answer shows which document and chunk it came from, along with a similarity score
- **Conversation memory** — The last 4 Q&A turns are passed to the LLM, enabling coherent follow-up questions
- **Multi-document support** — Upload multiple files in one session; the index grows incrementally without reloading the embedding model
- **Session management** — Clear just the conversation memory (keep documents) or wipe the entire session

---

## Architecture

```
User (Browser)
     │  HTTP
     ▼
Flask Backend (app.py)
     │
     ├── /upload ──► parser → chunker → embedder → FAISS index
     │
     ├── /ask ─────► retriever (top-k chunks)
     │                    │
     │               + history (last 4 turns)
     │                    │
     │                    ▼
     │               Gemini 2.5 Flash
     │                    │
     │               answer + sources
     │
     └── /clear_history   /clear
```

**RAG pipeline breakdown:**

| Step | Module | Detail |
|---|---|---|
| Parse | `rag/parser.py` | PyMuPDF for PDF, plain read for TXT |
| Chunk | `rag/chunker.py` | 400-word windows, 80-word overlap |
| Embed | `rag/embedder.py` | `all-MiniLM-L6-v2` via sentence-transformers |
| Index | `rag/embedder.py` | FAISS `IndexFlatL2`, incremental across uploads |
| Retrieve | `rag/retriever.py` | Top-4 nearest chunks by L2 distance |
| Generate | `rag/llm.py` | Gemini 2.5 Flash with source-labelled context + history |

---

## Project Structure

```
DocQuery/
├── app.py                  # Flask app — all routes
├── requirements.txt
├── .env                    # GEMINI_API_KEY goes here (not committed)
├── uploads/                # Temp storage for uploaded files
├── templates/
│   └── index.html          # Chat UI (HTML/CSS/JS, single file)
└── rag/
    ├── __init__.py
    ├── parser.py           # PDF/TXT → raw text
    ├── chunker.py          # text → overlapping chunk dicts
    ├── embedder.py         # chunks → embeddings → FAISS index
    ├── retriever.py        # query → top-k chunk dicts with scores
    └── llm.py              # chunks + history → Gemini answer
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/DocQuery.git
cd DocQuery
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Gemini API key**

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```
Get a free key at [aistudio.google.com](https://aistudio.google.com).

**5. Run**
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

1. **Upload a document** — drag and drop or click the upload zone in the sidebar (PDF or TXT)
2. **Ask a question** — type in the chat box and press Enter
3. **View sources** — each answer shows the source file, chunk ID, and similarity score
4. **Follow up** — ask follow-up questions naturally; the model remembers the last 4 turns
5. **Reset memory** — click *reset* in the Memory bar to clear history without re-uploading documents
6. **Clear session** — removes all documents and history

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | FAISS (`faiss-cpu`) |
| PDF parsing | PyMuPDF (`fitz`) |
| LLM | Gemini 2.5 Flash (Google GenAI) |
| Frontend | Vanilla HTML, CSS, JS |

---

## Possible Improvements

- [ ] Reranking with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for better retrieval quality
- [ ] Sentence-aware chunking using `nltk` or `spacy` instead of word-count windows
- [ ] Streaming responses via server-sent events
- [ ] Persistent FAISS index using `faiss.write_index()` so documents survive restarts
- [ ] Confidence thresholding — skip LLM call if retrieved chunks are below a similarity threshold
