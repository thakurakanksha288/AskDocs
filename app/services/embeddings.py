"""
Embedding service using Gemini text-embedding-004.

Design decisions
----------------
- Batching: the Gemini embedding API accepts up to 100 texts per request;
  we batch at embedding_batch_size (default 32) to stay within limits while
  reducing round-trips vs. one-at-a-time calls.
- Async: google-generativeai embed_content is synchronous; we wrap it in
  run_in_executor so it does not block the event loop.
- Caching: embeddings for a given text are cached in Redis (or in-memory
  fallback) keyed by sha256(text). This avoids re-embedding on re-upload.
"""

import asyncio
from typing import TYPE_CHECKING

import google.generativeai as genai

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import logger

if TYPE_CHECKING:
    from app.services.cache import CacheService

settings = get_settings()

genai.configure(api_key=settings.gemini_api_key)


def _embed_batch_sync(texts: list[str]) -> list[list[float]]:
    """Blocking call - runs in a thread executor."""
    result = genai.embed_content(
        model=settings.embedding_model,
        content=texts,
        task_type="RETRIEVAL_DOCUMENT",
    )
    embeddings = result["embedding"]
    if isinstance(embeddings[0], list):
        return embeddings
    return [embeddings]


async def embed_texts(
    texts: list[str],
    cache: "CacheService | None" = None,
) -> list[list[float]]:
    """
    Embed a list of texts, using cache when available.
    Returns embeddings in the same order as texts.
    """
    embeddings: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    # Cache lookup
    for i, text in enumerate(texts):
        if cache:
            cached = await cache.get_embedding(text)
            if cached is not None:
                embeddings[i] = cached
                continue
        uncached_indices.append(i)
        uncached_texts.append(text)

    if uncached_texts:
        loop = asyncio.get_event_loop()
        all_vecs: list[list[float]] = []

        batch_size = settings.embedding_batch_size
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start : batch_start + batch_size]
            try:
                vecs = await loop.run_in_executor(None, _embed_batch_sync, batch)
                all_vecs.extend(vecs)
            except Exception as exc:
                raise EmbeddingError(f"Embedding API error: {exc}") from exc

        for idx, vec in zip(uncached_indices, all_vecs, strict=True):
            embeddings[idx] = vec
            if cache:
                await cache.set_embedding(texts[idx], vec)

        logger.info("embeddings_fetched", count=len(uncached_texts))

    return embeddings  # type: ignore[return-value]


async def embed_query(
    text: str, cache: "CacheService | None" = None
) -> list[float]:
    """Embed a single query string with RETRIEVAL_QUERY task type."""
    loop = asyncio.get_event_loop()
    if cache:
        cached = await cache.get_embedding(text)
        if cached is not None:
            return cached
    try:
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model=settings.embedding_model,
                content=text,
                task_type="RETRIEVAL_QUERY",
            ),
        )
        vec: list[float] = result["embedding"]
    except Exception as exc:
        raise EmbeddingError(f"Query embedding error: {exc}") from exc

    if cache:
        await cache.set_embedding(text, vec)
    return vec
