"""
rag_engine.py — Embedding, indexing, and retrieval pipeline.

Framework-agnostic: no Streamlit imports. The caller (app.py) is responsible
for caching the embedding model with @st.cache_resource.

FAISS is used when available (faiss-cpu); otherwise falls back to pure NumPy
cosine-similarity. Both paths expose the same retrieve() interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from config import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD, EMBEDDING_MODEL_NAME

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Detect optional FAISS dependency at import time
try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
    logger.info("faiss-cpu is available — using IndexFlatIP backend.")
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not found — falling back to NumPy cosine similarity.")


# ── Model loading (framework-agnostic) ────────────────────────────────────


def load_embedding_model() -> SentenceTransformer:
    """Load and return the SentenceTransformer embedding model.

    This function is intentionally free of Streamlit imports so it can be
    used in testing or non-Streamlit contexts. In app.py it is wrapped with
    @st.cache_resource to avoid repeated loads.

    Returns:
        A loaded SentenceTransformer model instance.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ── Index construction ─────────────────────────────────────────────────────


def build_index(
    chunks: list[dict],
    model: SentenceTransformer,
) -> tuple[np.ndarray, list[dict]]:
    """Encode all chunks and return a normalised embedding matrix.

    L2-normalisation ensures dot-product == cosine similarity, which is
    required for both IndexFlatIP (FAISS) and the NumPy fallback.

    Args:
        chunks: List of chunk dicts (each must have a "text" key).
        model: Loaded SentenceTransformer model.

    Returns:
        A tuple of:
            - embedding_matrix (np.ndarray): shape (n_chunks, embedding_dim),
              dtype float32, L2-normalised row-wise.
            - chunks (list[dict]): The same list passed in, preserved in order
              so index positions match the matrix rows.
    """
    if not chunks:
        return np.empty((0,), dtype=np.float32), []

    texts = [c["text"] for c in chunks]
    logger.info("Encoding %d chunks…", len(texts))

    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # L2 normalise so dot product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)  # avoid division by zero
    embeddings = embeddings / norms

    return embeddings, chunks


# ── Retrieval ──────────────────────────────────────────────────────────────


def _retrieve_faiss(
    query_vec: np.ndarray,
    embedding_matrix: np.ndarray,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """FAISS-backed retrieval using IndexFlatIP (inner product on L2-normed vecs).

    Args:
        query_vec: 1-D normalised query embedding, shape (dim,).
        embedding_matrix: 2-D normalised chunk embeddings, shape (n, dim).
        chunks: Chunk dicts aligned with the matrix rows.
        top_k: Number of results to return.

    Returns:
        Top-k chunk dicts each augmented with a "similarity" float key.
    """
    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embedding_matrix)

    scores, indices = index.search(query_vec.reshape(1, -1), min(top_k, len(chunks)))
    scores = scores[0]
    indices = indices[0]

    results = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        chunk = dict(chunks[idx])
        chunk["similarity"] = float(np.clip(score, 0.0, 1.0))
        results.append(chunk)
    return results


def _retrieve_numpy(
    query_vec: np.ndarray,
    embedding_matrix: np.ndarray,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """Pure-NumPy cosine-similarity retrieval (fallback when FAISS unavailable).

    Args:
        query_vec: 1-D normalised query embedding, shape (dim,).
        embedding_matrix: 2-D normalised chunk embeddings, shape (n, dim).
        chunks: Chunk dicts aligned with the matrix rows.
        top_k: Number of results to return.

    Returns:
        Top-k chunk dicts each augmented with a "similarity" float key.
    """
    scores = embedding_matrix @ query_vec  # dot product == cosine (both normalised)
    top_indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]

    results = []
    for idx in top_indices:
        chunk = dict(chunks[idx])
        chunk["similarity"] = float(np.clip(scores[idx], 0.0, 1.0))
        results.append(chunk)
    return results


def retrieve(
    query: str,
    model: SentenceTransformer,
    embedding_matrix: np.ndarray,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a given query.

    Automatically selects FAISS or NumPy backend based on availability.

    Args:
        query: The user's question as a plain string.
        model: The SentenceTransformer model used for encoding.
        embedding_matrix: L2-normalised chunk embedding matrix from build_index().
        chunks: Chunk dicts aligned with the matrix rows.
        top_k: Number of chunks to return.

    Returns:
        A list of up to top_k chunk dicts, each augmented with "similarity"
        (float 0–1), sorted by descending similarity.
    """
    if embedding_matrix.shape[0] == 0 or not chunks:
        return []

    query_vec = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    query_vec = query_vec[0].astype(np.float32)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    if _FAISS_AVAILABLE:
        return _retrieve_faiss(query_vec, embedding_matrix, chunks, top_k)
    else:
        return _retrieve_numpy(query_vec, embedding_matrix, chunks, top_k)


# ── Confidence scoring ─────────────────────────────────────────────────────


def compute_confidence(retrieved: list[dict], grounded: bool) -> float:
    """Compute a composite confidence score for the generated answer.

    Formula:
        confidence = 0.6 * avg(similarity of retrieved chunks)
                   + 0.4 * (1.0 if grounded else 0.0)

    The score is clamped to [0.0, 1.0].

    Args:
        retrieved: List of retrieved chunk dicts with a "similarity" key.
        grounded: Whether the LLM reported GROUNDED: YES.

    Returns:
        A float in [0.0, 1.0].
    """
    if not retrieved:
        return 0.0

    avg_sim = float(np.mean([c.get("similarity", 0.0) for c in retrieved]))
    grounding_bonus = 1.0 if grounded else 0.0
    raw = 0.6 * avg_sim + 0.4 * grounding_bonus
    return float(np.clip(raw, 0.0, 1.0))
