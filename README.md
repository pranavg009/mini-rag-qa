# 🔎 Mini-RAG Document Q&A Bot

A production-quality Streamlit app that lets you upload documents (PDF, TXT, DOCX), index them locally with semantic embeddings, and ask questions — answered strictly from your document content via the Groq LLM API.

---

## Features

- **Local RAG pipeline** — no LangChain or LlamaIndex; built from scratch with `sentence-transformers` + FAISS
- **Grounded answers only** — the model is instructed to say "I don't know" rather than hallucinate
- **Confidence scoring** — composite score from retrieval similarity + grounding signal
- **Full chat history** — downloadable plain-text transcript
- **Clean error handling** — no raw tracebacks ever shown to the user

---

## Setup

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
The user's question is embedded with the same model. The top-K most similar chunks are retrieved via the index.

### 4. Grounded generation
Retrieved chunks are sent as numbered context to `llama-3.3-70b-versatile` on Groq with a strict system prompt: answer only from the provided chunks, cite them, and append `GROUNDED: YES` or `GROUNDED: NO`. The app strips this marker and uses it for scoring.

### 5. Confidence score
`confidence = 0.6 × avg_chunk_similarity + 0.4 × grounding_flag`  
Displayed as a colour-coded badge: 🟢 ≥75% · 🟠 50–74% · 🔴 <50%.

---

## File structure

```
mini-rag-qa/
├── app.py               # Streamlit UI
├── rag_engine.py        # Embedding, indexing, retrieval, confidence
├── document_loader.py   # Text extraction + chunking
├── llm_client.py        # Groq API client + answer generation
├── utils.py             # Badge formatter, transcript export, timer
├── config.py            # All constants (no magic numbers elsewhere)
├── requirements.txt
├── .streamlit/
│   └── config.toml      # Theme (indigo accent, auto light/dark)
├── .env.example
└── README.md
```
