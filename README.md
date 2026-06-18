# 🔎  Forensis — Mini-RAG Document Q&A Bot

**Live app:** [mini-rag-project.streamlit.app](https://mini-rag-project.streamlit.app)

A Streamlit app that lets you file documents (PDF, TXT, DOCX), index them locally with semantic embeddings, and pose questions — answered strictly from your document content via the Groq LLM API, with every reply measured for confidence and traced back to its source.

---

## Features

- **Local RAG pipeline** — no LangChain or LlamaIndex; built from scratch with `sentence-transformers` + FAISS
- **Grounded answers only** — the model is instructed to say "I don't know" rather than hallucinate
- **Confidence dial** — a composite score from retrieval similarity and a grounding signal, rendered as a measured reading rather than a generic badge
- **Full chat history** — downloadable plain-text transcript
- **Clean error handling** — no raw tracebacks ever shown to the user

---

## Try it now

The hosted version is live at **[mini-rag-project.streamlit.app](https://mini-rag-project.streamlit.app)** — no setup required. File a PDF, TXT, or DOCX from the sidebar, click **Catalog Documents**, then ask a question.

---

## Setup (local)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd mini-rag-qa
pip install -r requirements.txt
```

### 2. Set your Groq API key

Get a free key at https://console.groq.com

**Local development** — copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
# then edit .env:
GROQ_API_KEY=gsk_your_actual_key_here
```

**Streamlit Community Cloud** — go to your app's **Settings → Secrets** and add:

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

### 3. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## How it works

### 1. Document ingestion
Uploaded files are extracted to plain text (pdfplumber for PDF, python-docx for DOCX, UTF-8/latin-1 for TXT). Each document is split into overlapping character-level windows (800 chars, 150 overlap) that respect word boundaries.

### 2. Embedding & indexing
All chunks are encoded with `all-MiniLM-L6-v2` (a fast, high-quality local sentence embedding model) into L2-normalised vectors. These are stored in a FAISS `IndexFlatIP` (or a NumPy fallback) so dot-product search equals cosine similarity.

### 3. Retrieval
The question is embedded with the same model. The top-K most similar chunks are retrieved via the index, where K auto-scales with how many chunks are on file.

### 4. Grounded generation
Retrieved chunks are sent as numbered context to `llama-3.3-70b-versatile` on Groq with a strict system prompt: answer only from the provided chunks, never fabricate, and append `GROUNDED: YES` or `GROUNDED: NO`. The app strips this marker and uses it for scoring.

### 5. Confidence score
The raw cosine similarity from MiniLM clusters tightly (roughly 0.2–0.65 even for strong matches), so it's rescaled to a fuller 0–1 range before blending:

```
confidence = 0.45 × rescaled_avg_similarity
           + 0.20 × rescaled_top_chunk_similarity
           + 0.35 × grounding_flag
```

Displayed as a tick-gauge dial rather than a flat badge: 🟢 ≥68% high confidence · 🟠 42–67% medium · 🔴 <42% low.

---

## File structure

```
mini-rag-qa/
├── app.py               # Streamlit UI (Reading Room design)
├── rag_engine.py         # Embedding, indexing, retrieval, confidence
├── document_loader.py    # Text extraction + chunking
├── llm_client.py          # Groq API client + answer generation
├── utils.py               # Badge formatter, transcript export, timer
├── config.py              # All constants (no magic numbers elsewhere)
├── requirements.txt
├── .streamlit/
│   └── config.toml        # Theme (parchment & sepia, archival palette)
├── .env.example
└── README.md
```
