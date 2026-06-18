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
    page_title="Mini-RAG Q&A",
    page_icon="🔎",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Font stack ────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    }

    /* ── Answer card ────────────────────────────────────────────────────── */
    .answer-card {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border-left: 4px solid #6366f1;
        background: var(--background-secondary, rgba(99,102,241,0.06));
        margin-bottom: 0.75rem;
        line-height: 1.7;
    }

    /* ── Ungrounded answer — de-emphasised ────────────────────────────── */
    .answer-card.ungrounded {
        border-left-color: #f59e0b;
        background: var(--background-secondary, rgba(245,158,11,0.06));
        opacity: 0.88;
    }

    /* ── Confidence badge ───────────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-green  { background: rgba(34,197,94,0.18);  color: #15803d; }
    .badge-orange { background: rgba(245,158,11,0.18); color: #b45309; }
    .badge-red    { background: rgba(239,68,68,0.18);  color: #b91c1c; }

    /* ── Source chunk card ──────────────────────────────────────────────── */
    .source-card {
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: var(--background-secondary, rgba(100,100,100,0.06));
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        border: 1px solid rgba(100,100,100,0.12);
    }
    .source-meta {
        font-size: 0.78rem;
        opacity: 0.7;
        margin-bottom: 0.35rem;
    }

    /* ── History entry ──────────────────────────────────────────────────── */
    .history-entry {
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(100,100,100,0.12);
        background: var(--background-secondary, rgba(100,100,100,0.03));
    }
    .history-question {
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .history-meta {
        font-size: 0.76rem;
        opacity: 0.6;
        margin-top: 0.5rem;
    }

    /* ── Accent colour override for Streamlit buttons ────────────────────── */
    .stButton > button:first-child {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        st.title("🔎 Mini-RAG Q&A")
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
        top_k = st.slider(
            "Top-K chunks to retrieve",
            min_value=MIN_TOP_K,
            max_value=MAX_TOP_K,
            value=DEFAULT_TOP_K,
            help="More chunks give richer context but may add noise.",
        )

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
    card_class = "answer-card" if grounded else "answer-card ungrounded"
    st.markdown(f'<div class="{card_class}">{answer}</div>', unsafe_allow_html=True)

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
    st.title("🔎 Mini-RAG Document Q&A Bot")
    st.caption(
        "Upload documents, process them, and ask questions — "
        "answers are grounded strictly in the uploaded content."
    )

    # ── Trigger document processing ────────────────────────────────────────
    if process_btn:
        _process_documents(uploaded_files)

    # ── Status banner ──────────────────────────────────────────────────────
    if st.session_state["doc_ready"]:
        n_files = len(st.session_state["processed_files"])
        n_chunks = len(st.session_state["chunks"])
        st.info(
            f"📚 **{n_files} file(s)** · **{n_chunks} chunks** indexed and ready.",
            icon="✅",
        )
    else:
        st.info(
            "👆 Upload documents using the sidebar and click **Process Documents** to get started.",
            icon="ℹ️",
        )

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
