"""Health check endpoint - verifies Postgres, Redis and FAISS."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.cache import cache_service
from app.services.vector_store import vector_store

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    postgres: str
    redis: str
    vector_index: dict


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    # Postgres
    try:
        await db.execute(text("SELECT 1"))
        pg_status = "ok"
    except Exception as exc:
        pg_status = f"error: {exc}"

    # Redis
    redis_status = (
        "ok" if cache_service.is_available else "unavailable (using in-memory fallback)"
    )

    # FAISS
    vi = {"total_vectors": vector_store.total_vectors}

    overall = "ok" if pg_status == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        postgres=pg_status,
        redis=redis_status,
        vector_index=vi,
    )
