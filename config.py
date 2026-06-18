"""
config.py — Central configuration constants for Mini-RAG Q&A Bot.
All tunable parameters live here; no magic numbers in other modules.
"""

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 800          # characters per chunk
CHUNK_OVERLAP: int = 150       # overlap between consecutive chunks

# ── Models ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"

# ── Retrieval ─────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = 4
MIN_TOP_K: int = 1
MAX_TOP_K: int = 10

# ── Confidence thresholds ─────────────────────────────────────────────────
CONFIDENCE_HIGH_THRESHOLD: float = 0.75
CONFIDENCE_MEDIUM_THRESHOLD: float = 0.50

# ── File handling ─────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB: int = 25
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".docx")
