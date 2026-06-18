"""
llm_client.py — Groq API client initialisation and answer generation.

Reads GROQ_API_KEY from Streamlit secrets or environment variables.
All API errors are caught and re-raised as LLMGenerationError with
clean, user-facing messages.
"""

from __future__ import annotations

import logging
import os

import groq
import streamlit as st

from config import GROQ_MODEL_NAME

logger = logging.getLogger(__name__)


# ── Custom exceptions ──────────────────────────────────────────────────────


class MissingAPIKeyError(RuntimeError):
    """Raised when GROQ_API_KEY cannot be found in secrets or environment."""


class LLMGenerationError(RuntimeError):
    """Raised when the Groq API call fails for any reason."""


# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = """You are a precise document-analysis assistant. Answer the user's question using ONLY the information in the context chunks provided below. Follow these rules exactly:

1. If the context fully or partially supports an answer, answer clearly and concisely in plain language, a1. If the context fully or partially supports an answer, answer clearly and concisely in plain language. Do not add any inline citations or source references inside your answer text.nd cite which chunk(s) you used like this: (Source: Chunk 2).
2. If the context does NOT contain enough information to answer the question, respond with exactly: "I don't know based on the provided document(s)." followed by one short sentence on what information is missing. Do not guess. Do not use outside knowledge.
3. Never fabricate facts, numbers, or sources that are not in the context.
4. End your entire response with a new line containing exactly one of: GROUNDED: YES or GROUNDED: NO — YES only if your answer was fully supported by the provided context, NO if you said you don't know or had to guess at all."""


# ── Client initialisation ──────────────────────────────────────────────────


def get_groq_client() -> groq.Groq:
    """Initialise and return an authenticated Groq API client.

    Resolution order for the API key:
        1. st.secrets["GROQ_API_KEY"]  (Streamlit Cloud / local secrets.toml)
        2. os.environ["GROQ_API_KEY"]  (local .env loaded by python-dotenv)

    Returns:
        An authenticated groq.Groq client instance.

    Raises:
        MissingAPIKeyError: If no API key is found in either location.
    """
    api_key: str | None = None

    # ── 1. Try Streamlit secrets ───────────────────────────────────────────
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

    # ── 2. Fall back to environment variable ──────────────────────────────
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY not found. Please set it using one of these methods:\n\n"
            "• **Local development**: Create a `.env` file in the project root "
            "with the line:\n  `GROQ_API_KEY=your_groq_api_key_here`\n\n"
            "• **Streamlit Community Cloud**: Go to your app's Settings → Secrets "
            "and add:\n  `GROQ_API_KEY = \"your_groq_api_key_here\"`\n\n"
            "Get a free API key at https://console.groq.com"
        )

    return groq.Groq(api_key=api_key)


# ── Answer generation ──────────────────────────────────────────────────────


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    client: groq.Groq,
) -> dict:
    """Generate a grounded answer from the retrieved chunks using the Groq API.

    Builds a numbered-chunk context block, calls the model, strips the
    trailing GROUNDED: YES/NO marker, and returns structured results.

    Args:
        query: The user's question.
        retrieved_chunks: A list of chunk dicts (each must have "text" and "id").
        client: An authenticated groq.Groq client instance.

    Returns:
        A dict with keys:
            - "answer" (str): The LLM's answer, GROUNDED marker stripped.
            - "grounded" (bool): True if the model returned GROUNDED: YES.
            - "raw_response" (str): The full, unmodified model response.

    Raises:
        LLMGenerationError: For any API connectivity, rate-limit, or status error.
    """
    # ── Build numbered chunk context ───────────────────────────────────────
    context_lines: list[str] = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_lines.append(
            f"Chunk {i} (Source: {chunk.get('source', 'unknown')}, "
            f"Index: {chunk.get('chunk_index', '?')}):\n{chunk['text']}"
        )

    context_block = "\n\n---\n\n".join(context_lines)
    user_message = (
        f"CONTEXT CHUNKS:\n\n{context_block}\n\n"
        f"---\n\nQUESTION: {query}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # ── API call with specific exception handling ──────────────────────────
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
    except groq.APIConnectionError as exc:
        logger.error("Groq connection error: %s", exc)
        raise LLMGenerationError(
            "Could not connect to the Groq API. Please check your internet connection "
            "and try again."
        ) from exc
    except groq.RateLimitError as exc:
        logger.error("Groq rate limit hit: %s", exc)
        raise LLMGenerationError(
            "The Groq API rate limit has been reached. Please wait a moment and try again. "
            "If this persists, check your usage at https://console.groq.com"
        ) from exc
    except groq.APIStatusError as exc:
        logger.error("Groq API status error %s: %s", exc.status_code, exc.message)
        raise LLMGenerationError(
            f"The Groq API returned an error (HTTP {exc.status_code}): {exc.message}"
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error calling Groq API: %s", exc)
        raise LLMGenerationError(
            f"An unexpected error occurred while generating the answer: {exc}"
        ) from exc

    raw_response: str = response.choices[0].message.content or ""

    # ── Parse GROUNDED marker ──────────────────────────────────────────────
    grounded = False
    answer = raw_response.strip()

    lines = answer.splitlines()
    clean_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in ("GROUNDED: YES", "GROUNDED: NO"):
            grounded = stripped == "GROUNDED: YES"
        else:
            clean_lines.append(line)

    answer = "\n".join(clean_lines).strip()

    return {
        "answer": answer,
        "grounded": grounded,
        "raw_response": raw_response,
    }
