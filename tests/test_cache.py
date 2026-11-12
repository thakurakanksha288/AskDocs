"""Tests for the cache service (in-memory fallback path)."""
import pytest
from app.services.cache import CacheService


@pytest.fixture
def cache():
    svc = CacheService()
    svc._available = False  # force in-memory fallback
    return svc


@pytest.mark.asyncio
async def test_answer_cache_miss(cache):
    result = await cache.get_answer("What is X?", None)
    assert result is None


@pytest.mark.asyncio
async def test_answer_cache_hit(cache):
    payload = {"answer": "42", "sources": [], "latency_ms": 10.0}
    await cache.set_answer("What is X?", None, payload)
    result = await cache.get_answer("What is X?", None)
    assert result is not None
    assert result["answer"] == "42"


@pytest.mark.asyncio
async def test_different_doc_ids_different_keys(cache):
    payload = {"answer": "A", "sources": [], "latency_ms": 5.0}
    await cache.set_answer("Q?", ["doc1"], payload)
    result = await cache.get_answer("Q?", ["doc2"])
    assert result is None


@pytest.mark.asyncio
async def test_embedding_cache_roundtrip(cache):
    vec = [0.1, 0.2, 0.3]
    await cache.set_embedding("hello world", vec)
    result = await cache.get_embedding("hello world")
    assert result == vec


@pytest.mark.asyncio
async def test_embedding_cache_miss(cache):
    result = await cache.get_embedding("unknown text xyz")
    assert result is None
