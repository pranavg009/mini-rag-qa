"""
utils.py — Shared utility functions for the Mini-RAG Q&A Bot.

Contains UI formatting helpers, chat history export, and a timing utility.
"""

from __future__ import annotations

import time
from datetime import datetime
from functools import wraps
from typing import Callable

from config import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD


# ── Confidence badge ───────────────────────────────────────────────────────


def format_confidence_badge(confidence: float) -> tuple[str, str]:
    """Format a confidence float into a display label and a CSS colour name.

    Args:
        confidence: A float in [0.0, 1.0] representing answer confidence.

    Returns:
        A tuple of (label, color) where:
            - label is a human-readable string, e.g. "87% — High Confidence".
            - color is one of "green", "orange", or "red".
    """
    pct = int(round(confidence * 100))

    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        tier = "High Confidence"
        color = "green"
    elif confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        tier = "Medium Confidence"
        color = "orange"
    else:
        tier = "Low Confidence"
        color = "red"

    label = f"{pct}% — {tier}"
    return label, color


# ── Chat history export ────────────────────────────────────────────────────


def export_chat_history(history: list[dict]) -> str:
    """Serialise the session Q&A history to a plain-text transcript.

    Each entry in history must be a dict with keys:
        - "timestamp" (str | datetime): When the question was asked.
        - "question" (str): The user's question.
        - "answer" (str): The generated answer.
        - "confidence" (float): The confidence score for this answer.

    Args:
        history: A list of Q&A turn dicts from st.session_state["history"].

    Returns:
        A formatted plain-text transcript suitable for download as a .txt file.
    """
    if not history:
        return "No Q&A history to export."

    lines: list[str] = [
        "=" * 70,
        "  Mini-RAG Q&A Bot — Chat Transcript",
        f"  Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    for i, entry in enumerate(history, start=1):
        ts = entry.get("timestamp", "")
        if isinstance(ts, datetime):
            ts = ts.strftime("%Y-%m-%d %H:%M:%S")

        label, _ = format_confidence_badge(entry.get("confidence", 0.0))

        lines += [
            f"Turn {i}  [{ts}]",
            "-" * 50,
            f"Question: {entry.get('question', '')}",
            "",
            f"Answer:\n{entry.get('answer', '')}",
            "",
            f"Confidence: {label}",
            "",
        ]

    lines.append("=" * 70)
    return "\n".join(lines)


# ── Timing utility ─────────────────────────────────────────────────────────


class timed:
    """Context manager and decorator that measures elapsed wall-clock time.

    Usage as a context manager::

        with timed() as t:
            do_work()
        print(f"Took {t.elapsed:.2f}s")

    Usage as a decorator::

        @timed
        def my_func():
            ...
        result, elapsed = my_func()
    """

    def __init__(self, func: Callable | None = None) -> None:
        self._func = func
        self.elapsed: float = 0.0
        if func is not None:
            wraps(func)(self)

    # ── Context-manager interface ──────────────────────────────────────────

    def __enter__(self) -> "timed":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_args: object) -> None:
        self.elapsed = time.perf_counter() - self._start

    # ── Decorator interface ────────────────────────────────────────────────

    def __call__(self, *args: object, **kwargs: object):  # type: ignore[override]
        if self._func is None:
            raise TypeError("timed used as callable but no function was provided.")
        start = time.perf_counter()
        result = self._func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    def __get__(self, obj: object, objtype: object = None):  # type: ignore[override]
        """Support decoration of instance methods."""
        from functools import partial

        return partial(self.__call__, obj)
