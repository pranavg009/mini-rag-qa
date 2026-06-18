"""
app.py — Mini-RAG Document Q&A Bot (Streamlit entry point).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL_NAME,
    MAX_FILE_SIZE_MB,
    MAX_TOP_K,
    MIN_TOP_K,
)
from document_loader import (
    FileTooLargeError,
    UnsupportedFileTypeError,
    chunk_text,
    deduplicate_filename,
    load_document,
)
from llm_client import LLMGenerationError, MissingAPIKeyError, generate_answer, get_groq_client
from rag_engine import build_index, compute_confidence, load_embedding_model, retrieve
from utils import export_chat_history, format_confidence_badge, timed

# ── Bootstrap ─────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Forensis -- Mini_RAG Q&A",
    page_icon="🔎",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d14 0%, #111118 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ── Sidebar title accent ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] h1 {
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}

/* ── Main area background ─────────────────────────────────────────────── */
.main .block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* ── Page title gradient ──────────────────────────────────────────────── */
h1 {
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 60%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    letter-spacing: -1px;
    line-height: 1.2;
}

/* ── Section headings ─────────────────────────────────────────────────── */
h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.3px;
}

/* ── Answer card ──────────────────────────────────────────────────────── */
.answer-card {
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    background: linear-gradient(135deg,
        rgba(99,102,241,0.08) 0%,
        rgba(167,139,250,0.05) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366f1;
    margin-bottom: 1rem;
    line-height: 1.8;
    font-size: 1rem;
    box-shadow: 0 4px 24px rgba(99,102,241,0.08);
    backdrop-filter: blur(8px);
}

/* ── Ungrounded answer ────────────────────────────────────────────────── */
.answer-card.ungrounded {
    background: linear-gradient(135deg,
        rgba(245,158,11,0.08) 0%,
        rgba(251,191,36,0.04) 100%);
    border: 1px solid rgba(245,158,11,0.25);
    border-left: 4px solid #f59e0b;
    box-shadow: 0 4px 24px rgba(245,158,11,0.08);
}

/* ── Confidence badge ─────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-green {
    background: rgba(34,197,94,0.15);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.3);
    box-shadow: 0 0 12px rgba(34,197,94,0.15);
}
.badge-orange {
    background: rgba(245,158,11,0.15);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.3);
    box-shadow: 0 0 12px rgba(245,158,11,0.12);
}
.badge-red {
    background: rgba(239,68,68,0.15);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
    box-shadow: 0 0 12px rgba(239,68,68,0.12);
}

/* ── Metric cards ─────────────────────────────────────────────────────── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    background: linear-gradient(135deg,
        rgba(99,102,241,0.1) 0%,
        rgba(99,102,241,0.04) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    text-align: center;
}
.metric-card .metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.metric-card .metric-label {
    font-size: 0.75rem;
    opacity: 0.6;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}

/* ── Source chunk card ────────────────────────────────────────────────── */
.source-card {
    border-radius: 12px;
    padding: 1rem 1.2rem;
    background: rgba(255,255,255,0.03);
    margin-bottom: 0.75rem;
    font-size: 0.88rem;
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid #6366f1;
    transition: border-color 0.2s;
    line-height: 1.65;
}
.source-card:hover {
    border-left-color: #a78bfa;
    background: rgba(99,102,241,0.06);
}
.source-meta {
    font-size: 0.76rem;
    opacity: 0.55;
    margin-bottom: 0.5rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* ── History entry ────────────────────────────────────────────────────── */
.history-entry {
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.02);
    transition: border-color 0.2s;
}
.history-entry:hover {
    border-color: rgba(99,102,241,0.25);
}
.history-question {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
    color: #e2e8f0;
}
.history-meta {
    font-size: 0.74rem;
    opacity: 0.5;
    margin-top: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Chat input ───────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border-radius: 16px !important;
}
[data-testid="stChatInput"] textarea {
    border-radius: 14px !important;
    border: 1.5px solid rgba(99,102,241,0.4) !important;
    background: rgba(99,102,241,0.06) !important;
    font-size: 1rem !important;
    padding: 0.9rem 1.2rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.01em;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}

/* ── Alerts ───────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-width: 1px !important;
}

/* ── Expander ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    background: rgba(255,255,255,0.02) !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(99,102,241,0.3) !important;
}

/* ── Slider ───────────────────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #6366f1, #a78bfa) !important;
}

/* ── Dividers ─────────────────────────────────────────────────────────── */
hr {
    border-color: rgba(99,102,241,0.15) !important;
    margin: 1.5rem 0 !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99,102,241,0.3);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ───────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading embedding model…")
def _get_embedding_model():
    """Load the SentenceTransformer model, cached for the session lifetime."""
    return load_embedding_model()


@st.cache_resource(show_spinner="Initialising Groq client…")
def _get_groq_client():
    """Create and cache the Groq API client."""
    return get_groq_client()


# ── Session state initialisation ──────────────────────────────────────────


def _init_state() -> None:
    """Initialise all required session-state keys if not already present."""
    defaults = {
        "chunks": [],            # list[dict] — all processed chunks
        "embedding_matrix": None,  # np.ndarray | None
        "history": [],           # list[dict] — Q&A turns
        "processed_files": set(),  # set[str] — deduplicated filenames already ingested
        "doc_ready": False,      # bool — at least one doc processed
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────


def _render_sidebar() -> tuple[list, int]:
    """Render the sidebar and return (uploaded_files, top_k)."""
    with st.sidebar:
        st.title("🔎 Forensis -- Mini_RAG Q&A")
        st.caption("Upload documents, then ask questions grounded in their content.")
        st.divider()

        # ── File uploader ──────────────────────────────────────────────────
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx"],
            help=f"PDF, TXT, or DOCX · Max {MAX_FILE_SIZE_MB} MB each",
        )

        process_btn = st.button("⚙️ Process Documents", use_container_width=True, type="primary")

        st.divider()

        # ── Retrieval settings ─────────────────────────────────────────────
        st.subheader("Retrieval Settings")
        n = len(st.session_state.get("chunks", []))
        if n == 0:
            top_k = DEFAULT_TOP_K
        elif n <= 5:
            top_k = n
        elif n <= 15:
            top_k = max(3, n // 2)
        elif n <= 50:
            top_k = max(5, n // 5)
        else:
            top_k = min(MAX_TOP_K, max(7, n // 8))
        st.caption(f"🎯 Auto Top-K: **{top_k}** / {max(n,1)} chunks")

        st.divider()

        # ── Active model info ──────────────────────────────────────────────
        st.subheader("Active Models")
        st.markdown(
            f"**LLM:** `{GROQ_MODEL_NAME}`  \n"
            f"**Embeddings:** `{EMBEDDING_MODEL_NAME}`"
        )

        st.divider()

        # ── Reset session ──────────────────────────────────────────────────
        if st.button("🗑️ Reset Session", use_container_width=True):
            for key in ("chunks", "embedding_matrix", "history", "processed_files", "doc_ready"):
                if key in st.session_state:
                    del st.session_state[key]
            _init_state()
            st.rerun()

    return uploaded_files, top_k, process_btn


# ── Document processing ────────────────────────────────────────────────────


def _process_documents(uploaded_files: list) -> None:
    """Extract, chunk, and embed all uploaded files into session state."""
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one document before processing.")
        return

    model = _get_embedding_model()
    new_chunks: list[dict] = []
    errors: list[str] = []

    progress = st.progress(0, text="Processing documents…")
    total = len(uploaded_files)

    for i, file in enumerate(uploaded_files):
        progress.progress((i) / total, text=f"Processing {file.name}…")

        # ── Deduplicate filename ───────────────────────────────────────────
        unique_name = deduplicate_filename(file.name, st.session_state["processed_files"])

        try:
            raw_text = load_document(file)
        except UnsupportedFileTypeError as exc:
            errors.append(f"❌ **{file.name}**: {exc}")
            continue
        except FileTooLargeError as exc:
            errors.append(f"❌ **{file.name}**: {exc}")
            continue
        except Exception as exc:
            errors.append(f"❌ **{file.name}**: Unexpected error — {exc}")
            logger.exception("Error loading %s", file.name)
            continue

        if not raw_text or not raw_text.strip():
            errors.append(
                f"⚠️ **{file.name}**: No extractable text found. "
                "The file may be empty, image-only, or corrupted."
            )
            continue

        chunks = chunk_text(raw_text, source_name=unique_name)
        if not chunks:
            errors.append(f"⚠️ **{file.name}**: Text extracted but chunking produced no results.")
            continue

        new_chunks.extend(chunks)
        st.session_state["processed_files"].add(unique_name)

    progress.progress(1.0, text="Embedding chunks…")

    if new_chunks:
        all_chunks = st.session_state["chunks"] + new_chunks
        matrix, ordered_chunks = build_index(all_chunks, model)
        st.session_state["chunks"] = ordered_chunks
        st.session_state["embedding_matrix"] = matrix
        st.session_state["doc_ready"] = True

    progress.empty()

    for err in errors:
        st.error(err)

    if new_chunks:
        st.success(
            f"✅ Processed {len(st.session_state['processed_files'])} file(s) → "
            f"{len(st.session_state['chunks'])} total chunks indexed."
        )


# ── Q&A pipeline ───────────────────────────────────────────────────────────


def _run_qa(query: str, top_k: int) -> None:
    """Run retrieval + generation for a single query and render results."""
    model = _get_embedding_model()

    try:
        groq_client = _get_groq_client()
    except MissingAPIKeyError as exc:
        st.error(f"🔑 API Key Error\n\n{exc}")
        return

    embedding_matrix: np.ndarray = st.session_state["embedding_matrix"]
    chunks: list[dict] = st.session_state["chunks"]

    with st.spinner("Thinking…"):
        # ── Retrieval ──────────────────────────────────────────────────────
        timer = timed()
        with timer:
            retrieved = retrieve(query, model, embedding_matrix, chunks, top_k)

        # ── Generation ────────────────────────────────────────────────────
        try:
            gen_timer = timed()
            with gen_timer:
                result = generate_answer(query, retrieved, groq_client)
        except LLMGenerationError as exc:
            st.error(f"⚡ Generation Error\n\n{exc}")
            return

    total_elapsed = timer.elapsed + gen_timer.elapsed
    answer = result["answer"]
    grounded = result["grounded"]

    # ── Confidence ─────────────────────────────────────────────────────────
    confidence = compute_confidence(retrieved, grounded)
    badge_label, badge_color = format_confidence_badge(confidence)

    # ── Render answer ──────────────────────────────────────────────────────
    # Build unique source filenames used
    used_sources = list(dict.fromkeys([c.get("source", "unknown") for c in retrieved]))
    sources_html = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);'
        f'border-radius:999px;padding:2px 10px;font-size:0.76rem;font-weight:500;'
        f'margin-right:6px;">📄 {s}</span>'
        for s in used_sources
    )

    card_class = "answer-card" if grounded else "answer-card ungrounded"
    st.markdown(
        f'<div class="{card_class}">'
        f'{answer}'
        f'<div style="margin-top:1rem;padding-top:0.75rem;'
        f'border-top:1px solid rgba(255,255,255,0.08);">'
        f'<span style="font-size:0.74rem;opacity:0.5;'
        f'text-transform:uppercase;letter-spacing:0.05em;font-weight:600;">'
        f'Sources &nbsp;</span>{sources_html}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    # Badge + timing row
    col_badge, col_time, _ = st.columns([2, 2, 4])
    with col_badge:
        st.markdown(
            f'<span class="badge badge-{badge_color}">{badge_label}</span>',
            unsafe_allow_html=True,
        )
    with col_time:
        st.caption(f"⏱ Answered in {total_elapsed:.1f}s")

    if not grounded:
        st.warning(
            "⚠️ The model indicated the context was insufficient to fully answer this question. "
            "The answer above may be incomplete or speculative."
        )

    # ── Source expander ────────────────────────────────────────────────────
    with st.expander("📄 View Sources", expanded=False):
        if not retrieved:
            st.info("No chunks were retrieved.")
        else:
            for rank, chunk in enumerate(retrieved, start=1):
                sim_pct = int(round(chunk.get("similarity", 0.0) * 100))
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-meta">
                            #{rank} &nbsp;|&nbsp; <b>{chunk.get('source', 'unknown')}</b>
                            &nbsp;·&nbsp; Chunk {chunk.get('chunk_index', '?')}
                            &nbsp;·&nbsp; Similarity: <b>{sim_pct}%</b>
                        </div>
                        {chunk.get('text', '').replace(chr(10), '<br>')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Save to history ────────────────────────────────────────────────────
    st.session_state["history"].append(
        {
            "timestamp": datetime.now(),
            "question": query,
            "answer": answer,
            "confidence": confidence,
            "grounded": grounded,
            "retrieved": retrieved,
        }
    )


# ── History renderer ───────────────────────────────────────────────────────


def _render_history() -> None:
    """Render all previous Q&A turns in reverse-chronological order."""
    history = st.session_state["history"]
    if not history:
        return

    st.subheader("Previous Questions")

    for entry in reversed(history[:-1]):  # all turns except the current one
        label, color = format_confidence_badge(entry["confidence"])
        ts_str = ""
        if isinstance(entry.get("timestamp"), datetime):
            ts_str = entry["timestamp"].strftime("%H:%M:%S")

        st.markdown(
            f"""
            <div class="history-entry">
                <div class="history-question">🙋 {entry['question']}</div>
                <div>{entry['answer']}</div>
                <div class="history-meta">
                    <span class="badge badge-{color}">{label}</span>
                    &nbsp; {ts_str}
                    {'&nbsp; ⚠️ Not grounded' if not entry.get('grounded') else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Main layout ────────────────────────────────────────────────────────────


def main() -> None:
    """Top-level entry point — renders the full Streamlit UI."""

    uploaded_files, top_k, process_btn = _render_sidebar()

    # ── Header ─────────────────────────────────────────────────────────────
    st.title("🔎 Forensis -- Mini_RAG Document Q&A Bot")
    st.markdown(
        "<p style='font-size:1.05rem; opacity:0.75; margin-top:-0.5rem;'>"
        "Upload PDFs, TXT, or DOCX files · Ask questions · Get answers grounded "
        "strictly in your documents · Confidence-scored every time."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Trigger document processing ────────────────────────────────────────
    if process_btn:
        _process_documents(uploaded_files)

    # ── Status banner ──────────────────────────────────────────────────────
    if st.session_state["doc_ready"]:
        n_files = len(st.session_state["processed_files"])
        n_chunks = len(st.session_state["chunks"])
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{n_files}</div>
                <div class="metric-label">Documents Loaded</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{n_chunks}</div>
                <div class="metric-label">Chunks Indexed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{top_k}</div>
                <div class="metric-label">Auto Top-K Active</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            border-radius:16px;
            padding:1.5rem 2rem;
            background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(167,139,250,0.04));
            border:1px dashed rgba(99,102,241,0.3);
            text-align:center;
            margin:1rem 0;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">📂</div>
            <div style="font-weight:600;font-size:1.05rem;">No documents loaded yet</div>
            <div style="opacity:0.55;font-size:0.88rem;margin-top:0.3rem;">
                Upload PDF, TXT, or DOCX files from the sidebar and click Process Documents
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Question input ──────────────────────────────────────────────────────
    query = st.chat_input(
        "Ask a question about your documents…",
        disabled=not st.session_state["doc_ready"],
    )

    if query:
        query = query.strip()
        if not query:
            st.warning("⚠️ Please enter a non-empty question.")
        else:
            # Render the current question first
            st.markdown(f"**🙋 {query}**")
            st.markdown("")
            _run_qa(query, top_k)

    # ── History ────────────────────────────────────────────────────────────
    if len(st.session_state["history"]) > 1:
        st.divider()
        _render_history()

    # ── Download transcript ────────────────────────────────────────────────
    if st.session_state["history"]:
        transcript = export_chat_history(st.session_state["history"])
        st.download_button(
            label="⬇️ Download Transcript",
            data=transcript,
            file_name=f"rag_qa_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
