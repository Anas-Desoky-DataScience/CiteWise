"""CiteWise – RAG Pipeline.

Implements the full pipeline from the course notebooks:

  Lecture_3_RAG.ipynb
    • PyPDF text extraction → chunking → sentence-transformer embeddings → FAISS
  update-rag.ipynb
    • PyPDFLoader + CharacterTextSplitter(1000/100) + all-MiniLM-L6-v2 + FAISS
    • similarity_search_with_score for ranked retrieval
  Update_Chaining.ipynb
    • CustomHFLLM wrapping HF model inside a LangChain LLM
    • PromptTemplate → LLM chain
  Update_OutputParser (1).ipynb
    • StructuredOutputParser + ResponseSchema
    • extract_json_block helper for robust JSON extraction

Environment variables (optional overrides):
  EMBEDDING_MODEL  – HuggingFace embedding model name  (default: all-MiniLM-L6-v2)
  LLM_MODEL        – HuggingFace causal-LM model name  (default: Mistral-7B-Instruct-v0.2)
  LLM_BACKEND      – "local" | "extractive"            (default: local)
                     Use "extractive" for CPU-only / demo mode (no GPU needed).
  TOP_K            – number of chunks to retrieve      (default: 4)
  QUANTIZATION     – "4bit" | "none"                   (default: 4bit)
                     "4bit" loads the local model with BitsAndBytes NF4 quantization.
                     "none" loads the model in FP16 (requires more VRAM).
                     Has no effect when LLM_BACKEND="extractive".
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# ── Output-parser import (Update_OutputParser notebook style) ─────────────────
# Try langchain-classic first (the package used in the notebooks), fall back to
# the standard langchain path, then to langchain_community.
try:
    from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
except ImportError:
    try:
        from langchain.output_parsers import ResponseSchema, StructuredOutputParser
    except ImportError:
        from langchain_core.output_parsers import StrOutputParser  # type: ignore

        # Minimal stub so the rest of the code still runs
        class ResponseSchema:  # type: ignore
            def __init__(self, name: str, description: str):
                self.name = name
                self.description = description

        class StructuredOutputParser:  # type: ignore
            def __init__(self, schemas):
                self._schemas = schemas

            @classmethod
            def from_response_schemas(cls, schemas):
                return cls(schemas)

            def get_format_instructions(self) -> str:
                fields = ", ".join(f'"{s.name}"' for s in self._schemas)
                return f'Respond with a JSON object with keys: {fields}'

            def parse(self, text: str) -> dict:
                # Try JSON block first
                m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except json.JSONDecodeError:
                        pass
                # Try bare JSON object
                m2 = re.search(r"\{.*\}", text, re.DOTALL)
                if m2:
                    try:
                        return json.loads(m2.group(0))
                    except json.JSONDecodeError:
                        pass
                return {"answer": text.strip()}


# ── Configuration ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL       = os.getenv("LLM_MODEL",       "mistralai/Mistral-7B-Instruct-v0.2")
LLM_BACKEND     = os.getenv("LLM_BACKEND",     "local")   # "local" | "extractive"
TOP_K           = int(os.getenv("TOP_K",       "4"))
QUANTIZATION    = os.getenv("QUANTIZATION",    "4bit").lower()  # "4bit" | "none"

if QUANTIZATION not in ("4bit", "none"):
    raise ValueError(
        f"[CiteWise] Invalid QUANTIZATION value: {QUANTIZATION!r}. "
        "Expected '4bit' or 'none'."
    )

# ── Output Parser (from Update_OutputParser notebook) ─────────────────────────

_answer_schema = ResponseSchema(
    name="answer",
    description=(
        "A concise, well-reasoned answer to the student's question, "
        "using ONLY the provided context passages. "
        "If the answer cannot be found in the context, state that clearly."
    ),
)
answer_parser = StructuredOutputParser.from_response_schemas([_answer_schema])

# ── Prompt Template (from Update_Chaining notebook) ───────────────────────────

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are CiteWise, an academic AI assistant. "
        "Your job is to answer a student's question using ONLY the document passages provided below. "
        "Do not use any external knowledge. "
        "Write a clear, concise answer in plain prose — no JSON, no bullet symbols, no markdown. "
        "If the passages do not contain the answer, say: "
        "\"The document does not appear to contain information about this topic.\"\n\n"
        "Context (retrieved passages):\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)


# ── Custom HF LLM wrapper
# ─────────────────────

class CustomHFLLM(LLM):
    """Thin LangChain wrapper around a HuggingFace causal-LM (Mistral-style).

    Mirrors the `CustomHFLLM` class from Update_Chaining.ipynb, but adds:
    - lazy model loading (model downloaded only on first use)
    - chat-template support for instruction-tuned models
    """

    model_name: str = LLM_MODEL

    # Private runtime fields (excluded from Pydantic schema)
    _tokenizer: Any = None
    _model: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        print(f"[CiteWise] Loading LLM: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if QUANTIZATION == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
            )
            print("[CiteWise] LLM loaded with 4-bit NF4 quantization.")
        elif QUANTIZATION == "none":
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            print("[CiteWise] LLM loaded in FP16 (no quantization).")
        else:
            raise ValueError(
                f"[CiteWise] Unexpected QUANTIZATION value at load time: {QUANTIZATION!r}"
            )

    def _call(
        self,
        prompt: str,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:

        self._load()

        import torch

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # Return a dictionary containing input_ids + attention_mask
        model_inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move every tensor to the same device as the model
        model_inputs = {
            key: value.to(self._model.device)
            for key, value in model_inputs.items()
        }

        with torch.inference_mode():
            output_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.3,
            )

        # Decode only newly generated tokens
        prompt_length = model_inputs["input_ids"].shape[-1]

        new_tokens = output_ids[0][prompt_length:]

        return self._tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

    @property
    def _llm_type(self) -> str:
        return "citewise_hf"


# ── Singletons ─────────────────────────────────────────────────────────────────

_llm: CustomHFLLM | None = None
_embeddings: HuggingFaceEmbeddings | None = None


def get_llm() -> CustomHFLLM:
    global _llm
    if _llm is None:
        _llm = CustomHFLLM()
    return _llm


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print(f"[CiteWise] Loading embeddings: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("[CiteWise] Embeddings loaded.")
    return _embeddings


# ── JSON block extractor (from Update_OutputParser notebook) ──────────────────

def extract_json_block(text: str) -> str:
    """Pull the last ```json … ``` block from LLM output; fall back to first {…}.

    Mirrors the `extract_json_block` helper in Update_OutputParser (1).ipynb.
    """
    matches = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return f"```json\n{matches[-1]}\n```"
    m = re.search(r"\{.*\}", text, re.DOTALL)
    fallback = m.group(0) if m else json.dumps({"answer": text.strip()})
    return f"```json\n{fallback}\n```"


# ── Domain types ───────────────────────────────────────────────────────────────

@dataclass
class Source:
    page: int
    snippet: str
    score: float


@dataclass
class Document:
    name: str
    pages: int
    chunks: int
    vectordb: FAISS = field(repr=False)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def ingest_pdf(path: str, name: str) -> Document:
    """Load, chunk, embed, and index a PDF.

    Mirrors the ingestion steps from update-rag.ipynb:
        PyPDFLoader → CharacterTextSplitter(1000, 100) → HuggingFaceEmbeddings → FAISS
    """
    loader = PyPDFLoader(path)
    pages = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]

    if not chunks:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned/image PDF. Please use a text-based PDF."
        )

    vectordb = FAISS.from_documents(chunks, get_embeddings())
    return Document(name=name or "document.pdf", pages=len(pages), chunks=len(chunks), vectordb=vectordb)


def ask(doc: Document, question: str) -> tuple[str, list[Source]]:
    """Retrieve relevant chunks and generate an answer.

    Retrieval follows update-rag.ipynb (similarity_search_with_score, k=TOP_K).
    Answer generation follows Update_Chaining.ipynb (prompt → LLM chain).
    Parsing follows Update_OutputParser (1).ipynb (StructuredOutputParser + extract_json_block).
    """
    # ── 1. Retrieve top-k chunks (from update-rag.ipynb) ───────────────────
    hits = doc.vectordb.similarity_search_with_score(question, k=TOP_K)

    sources: list[Source] = []
    for chunk, distance in hits:
        page_num = int(chunk.metadata.get("page", 0)) + 1  # 0-indexed → 1-indexed
        snippet = " ".join(chunk.page_content.split())       # normalise whitespace
        # FAISS returns L2 distance; convert to a (0,1] similarity score
        similarity = float(1.0 / (1.0 + distance))
        sources.append(Source(page=page_num, snippet=snippet, score=round(similarity, 4)))

    if not sources:
        return "No relevant passages were found in the document for this question.", []

    # ── 2. Extractive fallback (no GPU / demo mode) ─────────────────────────
    if LLM_BACKEND == "extractive":
        best = sources[0]
        return (
            f"Most relevant passage (page {best.page}):\n\n{best.snippet}",
            sources,
        )

    # ── 3. Build context string ─────────────────────────────────────────────
    context = "\n\n".join(f"[Page {s.page}] {s.snippet}" for s in sources)

    # ── 4. Format prompt (Update_Chaining style) ───────────────────────────
    prompt_text = PROMPT.format(
        context=context,
        question=question,
    )

    # ── 5. Generate with LLM ────────────────────────────────────────────────
    llm = get_llm()
    answer_text = llm._call(prompt_text).strip()

    return answer_text, sources
