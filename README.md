# CiteWise — AI-Powered Academic Research Assistant

> Upload a PDF → Ask a question → Get a cited answer with page references.

CiteWise is a university graduation project that demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline for academic document question-answering.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Frontend (React)                   │
│   PDF Upload  →  Question Input  →  Answer + Sources │
└───────────────────────┬──────────────────────────────┘
                        │ HTTP (proxied by Vite /api/*)
                        ▼
┌──────────────────────────────────────────────────────┐
│              Backend (FastAPI · Python)               │
│  POST /api/upload    POST /api/ask    GET /api/health │
└───────────────────────┬──────────────────────────────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
    rag.ingest_pdf()          rag.ask()
    PyPDFLoader               FAISS similarity search
    CharacterTextSplitter     PromptTemplate → Mistral-7B
    FAISS.from_documents      StructuredOutputParser
    all-MiniLM-L6-v2          extract_json_block
```

### Notebooks → Code mapping

| Notebook | Feature used in `rag.py` |
|---|---|
| `Lecture_3_RAG.ipynb` | PDF extraction, chunking, embedding, FAISS index, retrieval |
| `update-rag.ipynb` | `PyPDFLoader` + `CharacterTextSplitter(1000,100)` + `HuggingFaceEmbeddings` + `FAISS` |
| `Update_Chaining.ipynb` | `CustomHFLLM` LangChain wrapper, `PromptTemplate` → LLM chain |
| `Update_OutputParser (1).ipynb` | `StructuredOutputParser` + `ResponseSchema` + `extract_json_block` |

---

## Quick Start

### 1 — Backend

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Install dependencies
cd backend
pip install -r requirements.txt

# Run the API server
uvicorn main:app --reload --port 8000
```

> **No GPU?** Set `LLM_BACKEND=extractive` to skip the Mistral model and use
> extractive answers (returns the most relevant passage directly):
> ```bash
> LLM_BACKEND=extractive uvicorn main:app --reload --port 8000
> ```

### 2 — Frontend

```bash
cd citewise---academic-study-\&-citation-assistant
npm install
npm run dev        # opens http://localhost:5173
```

The Vite dev server automatically proxies `/api/*` to `http://localhost:8000`.

---

## Environment Variables (Backend)

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Causal LM for answer generation |
| `LLM_BACKEND` | `local` | `local` (GPU HF model) or `extractive` (no LLM) |
| `TOP_K` | `4` | Number of document chunks to retrieve |

---

## API Reference

### `POST /api/upload`
Upload a PDF for indexing.

**Request:** `multipart/form-data` with field `file` (PDF)

**Response:**
```json
{
  "document_id": "abc123",
  "name": "lecture_notes.pdf",
  "pages": 42,
  "chunks": 138
}
```

### `POST /api/ask`
Ask a question against an uploaded document.

**Request:**
```json
{ "document_id": "abc123", "question": "What is gradient descent?" }
```

**Response:**
```json
{
  "answer": "Gradient descent is an optimization algorithm...",
  "sources": [
    { "page": 7, "snippet": "Gradient descent minimizes...", "score": 0.87 },
    { "page": 9, "snippet": "The learning rate controls...", "score": 0.81 }
  ]
}
```

### `GET /api/health`
Liveness check.

---

## Project Structure

```
Training Graduation Project/
├── backend/
│   ├── main.py             # FastAPI routes
│   ├── rag.py              # RAG pipeline (embeddings, FAISS, LLM, parser)
│   └── requirements.txt
│
├── citewise---academic-study-&-citation-assistant/
│   ├── src/
│   │   ├── App.tsx         # Main React UI (all three sections)
│   │   ├── index.css       # TailwindCSS v4 + custom animations
│   │   └── main.tsx        # React entry point
│   ├── index.html
│   ├── vite.config.ts      # Vite + /api proxy
│   └── package.json
│
├── Lecture_3_RAG.ipynb
├── Update_Chaining.ipynb
├── Update_OutputParser (1).ipynb
└── update-rag.ipynb
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, TailwindCSS v4, Vite 6, Lucide icons |
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (via LangChain HuggingFace) |
| Vector store | FAISS (CPU) |
| LLM | `mistralai/Mistral-7B-Instruct-v0.2` (via HuggingFace Transformers) |
| RAG framework | LangChain (core, community, text-splitters, huggingface) |
| PDF loading | PyPDFLoader (LangChain community) |
| Output parsing | LangChain `StructuredOutputParser` + `ResponseSchema` |
# CiteWise
# CiteWise
# CiteWise
# CiteWise
# CiteWise
