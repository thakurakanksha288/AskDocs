"""
Text extraction and chunking pipeline.

Design decisions
----------------
- Chunk size: ~500 tokens, overlap: ~50 tokens.
  * 500 tokens fits in one embedding call without truncation for most models,
    while giving enough context for a meaningful answer.
  * 50-token overlap prevents facts that straddle a boundary from disappearing.
  * Both are configurable via Settings so you can tune without a code change.
- PDF parsing runs in a thread-pool executor because PyMuPDF's fitz is a
  blocking C library. Running it in asyncio's event loop would stall all
  concurrent requests.
- For many large PDFs, we use a ProcessPoolExecutor so CPU-bound parsing
  runs in parallel across cores (bypasses the GIL).
"""

import asyncio
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import tiktoken

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()

# Use cl100k_base tokenizer (same family as GPT-4 / Gemini approximation).
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# A module-level process pool for CPU-heavy PDF work.
_PROCESS_POOL = ProcessPoolExecutor(max_workers=2)


# ── Extraction ────────────────────────────────────────────────────────────────


def _extract_pdf_sync(data: bytes) -> str:
    """CPU-bound: extract all text from a PDF byte string (runs in a worker)."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


async def extract_text(filename: str, data: bytes) -> str:
    """Return plain text from a file, dispatching blocking work off-thread."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        loop = asyncio.get_event_loop()
        # Heavy PDFs -> process pool so the GIL does not block the event loop.
        text: str = await loop.run_in_executor(
            _PROCESS_POOL, _extract_pdf_sync, data
        )
    elif suffix in {".txt", ".md"}:
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """
    Split text into overlapping token windows.

    Returns a list of dicts: {chunk_index, text, token_count}.
    """
    size = chunk_size or settings.chunk_size_tokens
    ovlp = overlap or settings.chunk_overlap_tokens

    if not text.strip():
        return []

    tokens = _TOKENIZER.encode(text)
    chunks: list[dict] = []
    start = 0

    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = _TOKENIZER.decode(chunk_tokens)
        chunks.append(
            {
                "chunk_index": len(chunks),
                "text": chunk_text_str,
                "token_count": len(chunk_tokens),
            }
        )
        if end == len(tokens):
            break
        start += size - ovlp  # slide forward with overlap

    logger.info(
        "text_chunked",
        total_tokens=len(tokens),
        num_chunks=len(chunks),
        chunk_size=size,
        overlap=ovlp,
    )
    return chunks
