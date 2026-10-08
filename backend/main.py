"""CiteWise – FastAPI Backend.

Start the server:
    cd backend
    uvicorn main:app --reload --port 8000

The frontend Vite dev server proxies /api/* to this server automatically.
"""

import os
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import rag

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="CiteWise API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document store (document_id → Document)
DOCS: dict[str, rag.Document] = {}

# ── Request / response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    document_id: str
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty.")
        return v.strip()


class SourceOut(BaseModel):
    page: int
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class UploadResponse(BaseModel):
    document_id: str
    name: str
    pages: int
    chunks: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Quick liveness check used by the frontend to detect server availability."""
    return {"status": "ok", "backend": "LLM_BACKEND=" + rag.LLM_BACKEND}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Receive a PDF, parse & index it, return document metadata."""
    filename = file.filename or "document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Write to a temp file so PyPDFLoader can open it by path
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        doc = await run_in_threadpool(rag.ingest_pdf, tmp_path, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    doc_id = uuid.uuid4().hex
    DOCS[doc_id] = doc

    return UploadResponse(
        document_id=doc_id,
        name=doc.name,
        pages=doc.pages,
        chunks=doc.chunks,
    )


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(body: AskRequest):
    """Answer a question against a previously uploaded document."""
    doc = DOCS.get(body.document_id)
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Please upload your PDF again.",
        )

    try:
        answer_text, sources = await run_in_threadpool(rag.ask, doc, body.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}")

    return AskResponse(
        answer=answer_text,
        sources=[SourceOut(page=s.page, snippet=s.snippet, score=s.score) for s in sources],
    )
