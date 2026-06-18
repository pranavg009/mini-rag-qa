"""
document_loader.py — Text extraction, validation, and chunking utilities.

Supports PDF (pdfplumber), DOCX (python-docx), and plain-text files.
Custom exceptions are raised for unsupported types and oversized files.
"""

from __future__ import annotations

import io
import logging
import re
import warnings
from typing import BinaryIO

import pdfplumber
from docx import Document

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_FILE_SIZE_MB,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)

# ── Custom exceptions ──────────────────────────────────────────────────────


class UnsupportedFileTypeError(ValueError):
    """Raised when a file's extension is not in SUPPORTED_EXTENSIONS."""


class FileTooLargeError(ValueError):
    """Raised when an uploaded file exceeds MAX_FILE_SIZE_MB."""


# ── Extraction helpers ─────────────────────────────────────────────────────


def extract_text_from_pdf(file: BinaryIO) -> str:
    """Extract text from a PDF file using pdfplumber.

    Pages that yield no extractable text (e.g. scanned images) are skipped
    with a warning appended to the returned string so callers are informed
    without crashing.

    Args:
        file: A file-like object positioned at the start of the PDF.

    Returns:
        Extracted text as a single string with page separators.
    """
    pages_text: list[str] = []
    warnings_list: list[str] = []

    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(text)
                else:
                    msg = f"[WARNING: Page {i} yielded no extractable text — it may be a scanned image.]"
                    warnings_list.append(msg)
                    logger.warning("PDF page %d has no extractable text.", i)
            except Exception as exc:  # pylint: disable=broad-except
                msg = f"[WARNING: Page {i} could not be read: {exc}]"
                warnings_list.append(msg)
                logger.warning("Error reading PDF page %d: %s", i, exc)

    combined = "\n\n".join(pages_text)
    if warnings_list:
        combined = combined + "\n\n" + "\n".join(warnings_list)
    return combined


def extract_text_from_docx(file: BinaryIO) -> str:
    """Extract text from a DOCX file using python-docx.

    All paragraph text is joined with newlines. Tables and headers inside
    the document body are also captured at paragraph level.

    Args:
        file: A file-like object containing the DOCX data.

    Returns:
        Extracted text as a single string.
    """
    doc = Document(file)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_txt(file: BinaryIO) -> str:
    """Extract text from a plain-text file.

    Tries UTF-8 first; falls back to latin-1 on decode errors.

    Args:
        file: A file-like object containing text data.

    Returns:
        Decoded text as a string.
    """
    raw_bytes = file.read()
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed; falling back to latin-1.")
        return raw_bytes.decode("latin-1")


# ── Public dispatcher ──────────────────────────────────────────────────────


def load_document(file: BinaryIO) -> str:
    """Dispatch to the correct extractor based on the file's extension.

    Also validates file size before extraction.

    Args:
        file: An uploaded file object with a `.name` attribute and `.read()`.

    Returns:
        The extracted text as a string.

    Raises:
        UnsupportedFileTypeError: If the extension is not in SUPPORTED_EXTENSIONS.
        FileTooLargeError: If the file exceeds MAX_FILE_SIZE_MB.
    """
    name: str = getattr(file, "name", "unknown")
    lower_name = name.lower()

    # ── Extension check ────────────────────────────────────────────────────
    ext = ""
    for supported in SUPPORTED_EXTENSIONS:
        if lower_name.endswith(supported):
            ext = supported
            break
    if not ext:
        raise UnsupportedFileTypeError(
            f"'{name}' has an unsupported file type. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    # ── Size check ─────────────────────────────────────────────────────────
    data = file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise FileTooLargeError(
            f"'{name}' is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )
    file_buffer = io.BytesIO(data)

    # ── Dispatch ───────────────────────────────────────────────────────────
    if ext == ".pdf":
        return extract_text_from_pdf(file_buffer)
    elif ext == ".docx":
        return extract_text_from_docx(file_buffer)
    elif ext == ".txt":
        return extract_text_from_txt(file_buffer)
    else:
        raise UnsupportedFileTypeError(f"Unhandled extension '{ext}'.")


# ── Chunking ───────────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    source_name: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split text into overlapping chunks, respecting word boundaries.

    Uses a sliding-window approach where each window steps forward by
    (chunk_size - overlap) characters, and the cut point is adjusted
    backward to the nearest whitespace so words are never split.

    Args:
        text: The full document text to chunk.
        source_name: A human-readable identifier for the source file
                     (used in chunk IDs and metadata).
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of chunk dicts, each with keys:
            - id (str): Unique identifier, e.g. "report_chunk_0".
            - text (str): The chunk's text content.
            - source (str): The source filename.
            - chunk_index (int): Zero-based position in the chunk sequence.
    """
    if not text or not text.strip():
        return []

    # Build a sanitised prefix for IDs (alphanumeric + underscore)
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", source_name)

    step = max(1, chunk_size - overlap)
    chunks: list[dict] = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to find last whitespace within the window to avoid mid-word splits
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk_text_slice = text[start:end].strip()
        if chunk_text_slice:
            chunks.append(
                {
                    "id": f"{safe_name}_chunk_{idx}",
                    "text": chunk_text_slice,
                    "source": source_name,
                    "chunk_index": idx,
                }
            )
            idx += 1

        start += step

    return chunks


# ── Duplicate filename handling ────────────────────────────────────────────


def deduplicate_filename(name: str, existing_names: set[str]) -> str:
    """Return a unique filename by appending an index suffix if necessary.

    Args:
        name: The desired filename.
        existing_names: A set of filenames already in use.

    Returns:
        A filename that is not present in existing_names.
    """
    if name not in existing_names:
        return name

    base, _, ext = name.rpartition(".")
    if not base:
        base = name
        ext = ""
    else:
        ext = "." + ext

    counter = 1
    candidate = f"{base}_{counter}{ext}"
    while candidate in existing_names:
        counter += 1
        candidate = f"{base}_{counter}{ext}"
    return candidate
