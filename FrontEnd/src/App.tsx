import { useRef, useState, useEffect } from 'react';
import {
  BookOpenText,
  FileText,
  Loader2,
  CheckCircle2,
  AlertCircle,
  SendHorizonal,
  UploadCloud,
  BookMarked,
  ChevronDown,
  X,
} from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────

type UploadStatus = 'idle' | 'processing' | 'ready' | 'error';

interface DocMeta {
  document_id: string;
  name: string;
  pages: number;
  chunks: number;
}

interface Source {
  page: number;
  snippet: string;
  score: number;
}

interface AnswerResult {
  question: string;
  text: string;
  sources: Source[];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

async function readError(res: Response): Promise<string> {
  try {
    const j = await res.json();
    return j.detail ?? j.message ?? res.statusText;
  } catch {
    return res.statusText;
  }
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatusBadge({ status, message, docName }: { status: UploadStatus; message: string; docName?: string }) {
  if (status === 'idle') return null;

  const cfg = {
    processing: { bg: 'bg-indigo-50 border-indigo-100', text: 'text-indigo-700', icon: <Loader2 size={14} className="animate-spin shrink-0" /> },
    ready:      { bg: 'bg-emerald-50 border-emerald-100', text: 'text-emerald-700', icon: <CheckCircle2 size={14} className="shrink-0" /> },
    error:      { bg: 'bg-red-50 border-red-100', text: 'text-red-600', icon: <AlertCircle size={14} className="shrink-0" /> },
  }[status] ?? { bg: '', text: '', icon: null };

  return (
    <div className={`mt-4 flex items-center gap-2 rounded-xl border px-4 py-2.5 text-sm ${cfg.bg} ${cfg.text}`}>
      {cfg.icon}
      {docName && <FileText size={14} className="shrink-0 opacity-60" />}
      <span className="truncate">
        {docName && <span className="font-semibold mr-1">{docName}</span>}
        {message}
      </span>
    </div>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = source.snippet.length > 200;
  const display = (!expanded && isLong) ? source.snippet.slice(0, 200) + '…' : source.snippet;

  return (
    <li className="group rounded-2xl border border-slate-100 bg-slate-50/70 p-4 transition hover:border-indigo-100 hover:bg-indigo-50/30">
      {/* Header row */}
      <div className="mb-2.5 flex items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-100 px-2.5 py-0.5 text-xs font-semibold text-indigo-700">
          <BookMarked size={11} />
          Page {source.page}
        </span>
        <span className="ml-auto text-xs text-slate-400 tabular-nums">
          {Math.round(source.score * 100)}% relevance
        </span>
        <span className="text-xs font-medium text-slate-400">#{index + 1}</span>
      </div>

      {/* Snippet */}
      <p className="text-sm leading-relaxed text-slate-600">{display}</p>

      {isLong && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-1.5 flex items-center gap-0.5 text-xs text-indigo-500 hover:text-indigo-700"
        >
          {expanded ? 'Show less' : 'Show more'}
          <ChevronDown size={12} className={`transition-transform ${expanded ? 'rotate-180' : ''}`} />
        </button>
      )}
    </li>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  // Upload state
  const [doc, setDoc] = useState<DocMeta | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle');
  const [uploadMsg, setUploadMsg] = useState('');
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // Question state
  const [question, setQuestion] = useState('');
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState<AnswerResult | null>(null);
  const [askError, setAskError] = useState('');

  // Scroll to answer when it arrives
  const answerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (answer) answerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [answer]);

  // ── Upload handler ──────────────────────────────────────────────────────────
  async function handleUpload(file: File) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadStatus('error');
      setUploadMsg('Only PDF files are supported.');
      return;
    }
    setDoc(null);
    setAnswer(null);
    setAskError('');
    setUploadStatus('processing');
    setUploadMsg(`Indexing "${file.name}"…`);

    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/upload', { method: 'POST', body: form });
      if (!res.ok) throw new Error(await readError(res));
      const d: DocMeta = await res.json();
      setDoc(d);
      setUploadStatus('ready');
      setUploadMsg(`${d.pages} pages · ${d.chunks} passages indexed`);
    } catch (err) {
      setUploadStatus('error');
      setUploadMsg(err instanceof Error ? err.message : 'Upload failed. Please try again.');
    }
  }

  // ── Ask handler ─────────────────────────────────────────────────────────────
  async function handleAsk(e?: React.FormEvent) {
    e?.preventDefault();
    if (!doc || !question.trim() || asking) return;
    setAsking(true);
    setAskError('');
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: doc.document_id, question: question.trim() }),
      });
      if (!res.ok) throw new Error(await readError(res));
      const data = await res.json();
      setAnswer({ question: question.trim(), text: data.answer, sources: data.sources });
    } catch (err) {
      setAskError(err instanceof Error ? err.message : 'Something went wrong. Please try again.');
    } finally {
      setAsking(false);
    }
  }

  // ── Drag events ─────────────────────────────────────────────────────────────
  function onDragOver(e: React.DragEvent) { e.preventDefault(); setDragging(true); }
  function onDragLeave() { setDragging(false); }
  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleUpload(f);
  }

  const canAsk = uploadStatus === 'ready' && question.trim().length > 0 && !asking;

  // ─── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-indigo-50/20 to-white">

      {/* ── Top bar ── */}
      <header className="sticky top-0 z-10 border-b border-slate-100/80 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-3xl items-center gap-3 px-5 py-3.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600">
            <BookOpenText size={17} className="text-white" />
          </div>
          <div>
            <h1 className="text-base font-semibold leading-none tracking-tight text-slate-900">CiteWise</h1>
            <p className="text-[11px] text-slate-400 mt-0.5">AI-Powered Academic Research Assistant</p>
          </div>

          {/* Doc chip */}
          {doc && (
            <div className="ml-auto flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">
              <CheckCircle2 size={12} />
              <span className="max-w-[160px] truncate">{doc.name}</span>
              <button
                onClick={() => { setDoc(null); setUploadStatus('idle'); setUploadMsg(''); setAnswer(null); }}
                className="ml-0.5 rounded-full p-0.5 hover:bg-emerald-100"
                title="Remove document"
              >
                <X size={11} />
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-3xl space-y-5 px-5 py-8">

        {/* ─── Hero line ─── */}
        <div className="text-center pt-2 pb-1">
          <p className="text-2xl font-semibold text-slate-800 tracking-tight">
            Ask anything about your document
          </p>
          <p className="mt-1 text-sm text-slate-500">
            Upload a PDF · Ask a question · Get a cited answer
          </p>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════
            SECTION 1 — PDF UPLOADER
        ══════════════════════════════════════════════════════════════════════ */}
        <section className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm">
          <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-400">
            Step 1 · Upload PDF
          </p>

          {/* Drop zone */}
          <div
            onClick={() => fileRef.current?.click()}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            className={`
              relative cursor-pointer select-none rounded-xl border-2 border-dashed px-6 py-10 text-center
              transition-all duration-200
              ${dragging
                ? 'border-indigo-400 bg-indigo-50 scale-[1.01]'
                : uploadStatus === 'ready'
                  ? 'border-emerald-300 bg-emerald-50/40'
                  : 'border-slate-200 hover:border-indigo-300 hover:bg-indigo-50/40'
              }
            `}
          >
            <input
              ref={fileRef}
              type="file"
              accept="application/pdf"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleUpload(f);
                e.target.value = '';
              }}
            />

            {uploadStatus === 'ready' && doc ? (
              <>
                <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-emerald-100">
                  <FileText size={22} className="text-emerald-600" />
                </div>
                <p className="font-semibold text-slate-700">{doc.name}</p>
                <p className="mt-0.5 text-sm text-slate-400">
                  {doc.pages} pages · Click to replace
                </p>
              </>
            ) : (
              <>
                <div className={`mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full transition-colors ${
                  dragging ? 'bg-indigo-200' : 'bg-indigo-100'
                }`}>
                  <UploadCloud size={22} className={dragging ? 'text-indigo-700' : 'text-indigo-500'} />
                </div>
                <p className="font-semibold text-slate-700">
                  {dragging ? 'Drop it here!' : 'Drag & drop your PDF here'}
                </p>
                <p className="mt-1 text-sm text-slate-400">or click to browse — lecture notes, papers, textbooks</p>
              </>
            )}
          </div>

          {/* Status message */}
          <StatusBadge
            status={uploadStatus}
            message={uploadMsg}
            docName={uploadStatus === 'ready' && doc ? undefined : undefined}
          />
        </section>

        {/* ═══════════════════════════════════════════════════════════════════
            SECTION 2 — QUESTION INPUT
        ══════════════════════════════════════════════════════════════════════ */}
        <section className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm">
          <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-400">
            Step 2 · Ask a Question
          </p>

          <form onSubmit={handleAsk} className="flex gap-2.5">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAsk(); }
              }}
              disabled={uploadStatus !== 'ready'}
              placeholder={
                uploadStatus === 'ready'
                  ? 'e.g. What is the main argument of chapter 3?'
                  : 'Upload a PDF first to enable questions…'
              }
              className="
                flex-1 rounded-xl border border-slate-200 bg-slate-50/60
                px-4 py-3 text-sm outline-none
                transition-all duration-150
                placeholder:text-slate-400
                focus:border-indigo-400 focus:bg-white focus:ring-4 focus:ring-indigo-100
                disabled:cursor-not-allowed disabled:opacity-50
              "
            />
            <button
              type="submit"
              disabled={!canAsk}
              className="
                inline-flex shrink-0 items-center gap-2 rounded-xl
                bg-indigo-600 px-5 py-3 text-sm font-semibold text-white
                shadow-sm shadow-indigo-200
                transition-all duration-150
                hover:bg-indigo-700 hover:shadow-indigo-300
                disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none
              "
            >
              {asking
                ? <><Loader2 size={15} className="animate-spin" />Thinking…</>
                : <><SendHorizonal size={15} />Ask</>
              }
            </button>
          </form>

          {askError && (
            <p className="mt-3 flex items-center gap-1.5 text-sm text-red-600">
              <AlertCircle size={14} className="shrink-0" />
              {askError}
            </p>
          )}
        </section>

        {/* ═══════════════════════════════════════════════════════════════════
            SECTION 3 — ANSWER + SOURCES
        ══════════════════════════════════════════════════════════════════════ */}
        {(asking || answer) && (
          <section ref={answerRef} className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm">
            <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-slate-400">
              Step 3 · AI Answer
            </p>

            {/* Loading state */}
            {asking && !answer && (
              <div className="flex flex-col items-center gap-3 py-10 text-slate-400">
                <div className="flex h-14 w-14 items-center justify-center rounded-full bg-indigo-50">
                  <Loader2 size={26} className="animate-spin text-indigo-500" />
                </div>
                <p className="text-sm">Searching the document and composing an answer…</p>
              </div>
            )}

            {/* Answer */}
            {answer && (
              <div className={`space-y-6 transition-opacity duration-300 ${asking ? 'opacity-40 pointer-events-none' : 'opacity-100'}`}>

                {/* Question echo */}
                <div className="rounded-xl bg-indigo-50/70 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-widest text-indigo-400 mb-1">Your question</p>
                  <p className="text-sm text-slate-700">"{answer.question}"</p>
                </div>

                {/* Answer text */}
                <div>
                  <p className="mb-2 text-[11px] font-semibold uppercase tracking-widest text-slate-400">Answer</p>
                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-800">
                    {answer.text}
                  </p>
                </div>

                {/* Sources */}
                {answer.sources.length > 0 && (
                  <div>
                    <p className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-slate-400">
                      Sources · {answer.sources.length} passage{answer.sources.length !== 1 ? 's' : ''} found
                    </p>
                    <ul className="space-y-2.5">
                      {answer.sources.map((s, i) => (
                        <SourceCard key={i} source={s} index={i} />
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </section>
        )}

        {/* Footer */}
        <footer className="pb-6 text-center text-xs text-slate-400">
          CiteWise · Powered by RAG + Mistral · University Graduation Project
        </footer>
      </main>
    </div>
  );
}
