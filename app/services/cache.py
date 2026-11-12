"""
Two-layer cache: Redis (primary) + in-memory dict (fallback).

Strategy
--------
- Full answers are cached keyed by sha256(question + sorted doc ids), TTL 1 h.
- Embeddings are cached keyed by sha256(text), TTL 24 h.
- If Redis is unavailable the service degrades gracefully to in-memory dicts.
- Cache is invalidated when a document is deleted (all answer keys that
  reference that document id are removed via Redis SCAN).
"""

import hashlib
import json
import pickle
from typing import Any

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()

_ANSWER_PREFIX = "ask:answer:"
_EMBED_PREFIX = "ask:emb:"

# In-memory fallback caches
_answer_fallback: dict[str, Any] = {}
_embed_fallback: dict[str, list[float]] = {}


def _answer_key(question: str, doc_ids: list[str] | None) -> str:
    raw = question + "|" + ",".join(sorted(doc_ids or []))
    return _ANSWER_PREFIX + hashlib.sha256(raw.encode()).hexdigest()


def _embed_key(text: str) -> str:
    return _EMBED_PREFIX + hashlib.sha256(text.encode()).hexdigest()


class CacheService:
    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None
        self._available = False

    async def connect(self, url: str) -> None:
        try:
            self._redis = aioredis.from_url(url, decode_responses=False)
            await self._redis.ping()
            self._available = True
            logger.info("redis_connected", url=url)
        except Exception as exc:
            self._available = False
            logger.warning("redis_unavailable", error=str(exc), fallback="in-memory")

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    # Answer cache

    async def get_answer(
        self, question: str, doc_ids: list[str] | None
    ) -> dict | None:
        key = _answer_key(question, doc_ids)
        if self._available and self._redis:
            try:
                raw = await self._redis.get(key)
                if raw:
                    logger.info("cache_hit", layer="redis", key=key[:20])
                    return json.loads(raw)
            except Exception:
                pass
        result = _answer_fallback.get(key)
        if result:
            logger.info("cache_hit", layer="memory", key=key[:20])
        return result

    async def set_answer(
        self, question: str, doc_ids: list[str] | None, payload: dict
    ) -> None:
        key = _answer_key(question, doc_ids)
        _answer_fallback[key] = payload
        if self._available and self._redis:
            try:
                await self._redis.setex(
                    key, settings.cache_ttl_seconds, json.dumps(payload)
                )
            except Exception as exc:
                logger.warning("redis_set_failed", error=str(exc))

    async def invalidate_by_doc_ids(self, doc_ids: list[str]) -> None:
        """Remove all answer cache entries that contain any of the given doc ids."""
        for doc_id in doc_ids:
            _answer_fallback.clear()
            if self._available and self._redis:
                try:
                    async for key in self._redis.scan_iter(f"{_ANSWER_PREFIX}*"):
                        raw = await self._redis.get(key)
                        if raw and doc_id.encode() in raw:
                            await self._redis.delete(key)
                except Exception as exc:
                    logger.warning("redis_invalidate_failed", error=str(exc))

    # Embedding cache

    async def get_embedding(self, text: str) -> list[float] | None:
        key = _embed_key(text)
        if self._available and self._redis:
            try:
                raw = await self._redis.get(key)
                if raw:
                    return pickle.loads(raw)  # noqa: S301
            except Exception:
                pass
        return _embed_fallback.get(key)

    async def set_embedding(self, text: str, vec: list[float]) -> None:
        key = _embed_key(text)
        _embed_fallback[key] = vec
        if self._available and self._redis:
            try:
                await self._redis.setex(
                    key, settings.embedding_cache_ttl, pickle.dumps(vec)
                )
            except Exception as exc:
                logger.warning("redis_set_embed_failed", error=str(exc))

    @property
    def is_available(self) -> bool:
        return self._available


# Singleton
cache_service = CacheService()
