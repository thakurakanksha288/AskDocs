"""
AskDocs FastAPI application entry point.

asyncio vs threading vs multiprocessing - which tool for which problem
----------------------------------------------------------------------
- asyncio handles all I/O-bound concurrency: HTTP requests, Postgres queries
  via asyncpg, Redis operations, and Gemini API calls. Thousands of concurrent
  requests share one event loop with no thread overhead.
- threading (run_in_executor) is used for blocking C-library calls that cannot
  be made async, such as PyMuPDF PDF parsing and the synchronous google-
  generativeai SDK. A thread releases the GIL during I/O, so multiple threads
  can overlap without blocking the event loop.
- multiprocessing is used for CPU-bound PDF extraction of large files
  (ProcessPoolExecutor in ingestion.py). The GIL prevents true parallelism in
  threads for CPU work, so we fork worker processes. Benchmarking showed a
  ~1.8x speedup when processing 4 large PDFs simultaneously (see README).
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import ask, docs_router, health, upload
from app.core.config import get_settings
from app.core.exceptions import AskDocsError
from app.core.logging import new_request_id, setup_logging
from app.db.models import Base
from app.db.session import engine
from app.services.cache import cache_service
from app.services.vector_store import vector_store

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging(settings.log_level)
    # Connect to Redis (non-fatal if unavailable)
    await cache_service.connect(settings.redis_url)
    # Load FAISS index from disk
    vector_store.load()
    # Create Postgres tables (idempotent)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown
    await cache_service.close()


app = FastAPI(
    title="AskDocs",
    description=(
        "Production-style RAG service: upload documents, ask questions, "
        "get cited answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = new_request_id()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(AskDocsError)
async def askdocs_error_handler(request: Request, exc: AskDocsError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# Routers
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(docs_router.router)
app.include_router(ask.router)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "AskDocs API - visit /docs for the interactive UI."}
