"""
/ask endpoint - the core RAG query pipeline.

Pipeline
--------
1. Check answer cache.
2. Embed the question (with embedding cache).
3. Similarity search in FAISS.
4. Load matching chunks from Postgres.
5. Call Gemini to generate a cited answer.
6. Log query to Postgres, store in cache, return.
"""
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import logger
from app.db.models import Chunk, Query
from app.db.session import get_db
from app.services.cache import cache_service
from app.services.embeddings import embed_query
from app.services.llm import generate_answer
from app.services.vector_store import vector_store

settings = get_settings()
router = APIRouter(prefix="/ask", tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict search to specific document ids."
    )
    top_k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    cached: bool
    latency_ms: float


@router.post("", response_model=AskResponse)
async def ask(req: AskRequest, db: AsyncSession = Depends(get_db)) -> AskResponse:
    t0 = time.perf_counter()

    # 1. Cache check
    cached_payload = await cache_service.get_answer(req.question, req.document_ids)
    if cached_payload:
        return AskResponse(**cached_payload, cached=True)

    # 2. Embed question
    q_vec = await embed_query(req.question, cache=cache_service)

    # 3. FAISS search
    raw_results = await vector_store.search(q_vec, top_k=req.top_k)
    if not raw_results:
        return AskResponse(
            answer="I don't know based on the provided documents.",
            sources=[],
            cached=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    chunk_id_to_score = dict(raw_results)

    # 4. Load chunks from Postgres
    stmt = select(Chunk).where(Chunk.id.in_(list(chunk_id_to_score.keys())))
    if req.document_ids:
        stmt = stmt.where(Chunk.document_id.in_(req.document_ids))
    result = await db.execute(stmt)
    chunks_db = result.scalars().all()

    if not chunks_db:
        return AskResponse(
            answer="I don't know based on the provided documents.",
            sources=[],
            cached=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # Sort by score descending
    chunks_db_sorted = sorted(
        chunks_db, key=lambda c: chunk_id_to_score.get(c.id, 0), reverse=True
    )
    chunk_dicts = [{"id": c.id, "text": c.text} for c in chunks_db_sorted]

    # 5. Generate answer
    answer = await generate_answer(req.question, chunk_dicts)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    sources = [
        SourceChunk(
            chunk_id=c.id,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            text=c.text[:400],  # truncate for response size
            score=round(chunk_id_to_score.get(c.id, 0), 4),
        )
        for c in chunks_db_sorted
    ]

    # 6. Cache the answer
    payload = {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "latency_ms": latency_ms,
    }
    await cache_service.set_answer(req.question, req.document_ids, payload)

    # 7. Log to Postgres
    db.add(
        Query(
            id=str(uuid.uuid4()),
            question=req.question,
            answer=answer,
            latency_ms=latency_ms,
            cache_hit=False,
        )
    )
    await db.commit()

    logger.info("ask_served", latency_ms=latency_ms, cached=False, sources=len(sources))
    return AskResponse(answer=answer, sources=sources, cached=False, latency_ms=latency_ms)
