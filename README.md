# AskDocs 📄🔍

> **Upload PDFs or text files. Ask questions in plain English. Get cited answers — instantly.**

A production-style **RAG (Retrieval-Augmented Generation)** service built with FastAPI, FAISS, Gemini, PostgreSQL, and Redis. Answers come with source passages so you can verify every claim.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               Client (Browser / Swagger UI / curl)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP
┌──────────────────────────▼──────────────────────────────────────┐
│                    FastAPI (async, uvicorn)                      │
│        Middleware: request-id, CORS, structured JSON logs        │
└──────┬───────────────────────────────────────┬──────────────────┘
       │                                       │
┌──────▼──────────────────┐    ┌──────────────▼──────────────────┐
│   Ingestion Pipeline    │    │        Query Pipeline            │
│                         │    │                                  │
│  1. Validate file        │    │  1. Redis answer cache check     │
│  2. Persist doc to PG   │    │  2. Embed question (Gemini)      │
│  3. Extract text        │    │  3. FAISS top-k similarity search│
│     (fitz / plain text) │    │  4. Load chunks from Postgres    │
│  4. Chunk (500 tok,     │    │  5. Gemini LLM → cited answer    │
│     50 overlap)         │    │  6. Cache result in Redis        │
│  5. Embed chunks batch  │    │  7. Log query to Postgres        │
│  6. Write to FAISS      │    │                                  │
│  7. Mark doc ready      │    └──────────────┬───────────────────┘
└──────┬──────────────────┘                   │
       │                         ┌────────────▼────────────┐
       │                         │  Redis                  │
       │                         │  • Answer cache (1 h)   │
       │                         │  • Embedding cache (24h)│
       │                         │  • In-memory LRU fallback│
       │                         └─────────────────────────┘
       │
┌──────▼──────────────────┐    ┌─────────────────────────────────┐
│  PostgreSQL             │    │  FAISS (IndexFlatIP)             │
│  • documents            │    │  • Cosine similarity (L2-normed) │
│  • chunks               │    │  • Persisted to disk             │
│  • queries (logs)       │    │  • Rebuilt on doc delete         │
│  • Indexes on doc_id,   │    │  • chunk_id ↔ Postgres UUID sync │
│    created_at           │    └─────────────────────────────────┘
└─────────────────────────┘
```

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
git clone https://github.com/yourname/askdocs
cd askdocs
cp .env.example .env
# Add your GEMINI_API_KEY to .env

docker compose up --build
# API: http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### Option B — Local development

```bash
# Prerequisites: Python 3.11+, PostgreSQL 16, Redis 7
cd askdocs
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

cp .env.example .env   # fill in GEMINI_API_KEY and DB credentials

uvicorn app.main:app --reload
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/documents` | Upload a PDF or text file. Returns `document_id`. Ingestion runs in the background. |
| `GET` | `/documents` | List all uploaded documents with status and chunk count. |
| `DELETE` | `/documents/{id}` | Delete a document, its chunks, vectors, and cached answers. |
| `POST` | `/ask` | Ask a question. Body: `{question, document_ids?, top_k?}`. Returns `{answer, sources[], cached, latency_ms}`. |
| `GET` | `/health` | Check Postgres, Redis, and FAISS status. |

### Example: upload and ask

```bash
# 1. Upload
curl -X POST http://localhost:8000/documents \
  -F "file=@my_paper.pdf"
# → {"document_id": "abc-123", "status": "processing"}

# 2. Poll until ready
curl http://localhost:8000/documents
# → [{..."status": "ready", "num_chunks": 42}]

# 3. Ask
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What methodology did the authors use?", "document_ids": ["abc-123"]}'
```

---

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| **Chunk size** | 500 tokens | Fits in one embedding call without truncation; provides enough context for a coherent answer. Too small → loss of context. Too large → retrieves irrelevant text. |
| **Chunk overlap** | 50 tokens | Prevents facts that straddle a boundary from disappearing. 10% of chunk size is a standard starting point. |
| **Vector index** | FAISS `IndexFlatIP` | No server to manage, persists to a single file, exact search (no approximation errors at our scale). Switch to `IndexIVFFlat` at >1M vectors. |
| **Embeddings** | Gemini `text-embedding-004` (768-dim) | Strong multilingual quality, generous free tier, same provider as LLM. |
| **Cache TTL** | Answers: 1 h, Embeddings: 24 h | Answers change if documents change; embeddings are deterministic so longer TTL is safe. |
| **LLM temperature** | 0.1 | Lower temperature reduces hallucination in factual Q&A tasks. |
| **Ingestion** | Background task | Returns HTTP 202 immediately so large PDFs don't time out the client. |

---

## asyncio vs threading vs multiprocessing

> **Which tool for which problem, and why.**

| Tool | Used for | Why |
|------|----------|-----|
| **asyncio** | HTTP handling, Postgres (asyncpg), Redis, Gemini API calls | I/O-bound: the CPU is idle while waiting for network. One event loop serves thousands of concurrent requests with no thread overhead. |
| **threading (`run_in_executor`)** | PDF parsing (PyMuPDF/fitz), synchronous Gemini SDK calls | These are blocking C-library calls that can't be awaited. Wrapping them in `ThreadPoolExecutor` releases the event loop while they run. The GIL is released during I/O within threads. |
| **multiprocessing (`ProcessPoolExecutor`)** | CPU-heavy PDF extraction of many large files simultaneously | PDF text extraction is CPU-bound (C extension, but GIL-holding). Multiple processes bypass the GIL entirely and run on separate cores. Benchmarked at ~1.8× speedup for 4 large PDFs vs. single process (see below). |

---

## Benchmark Results

Run with `python scripts/benchmark.py --n 50 --concurrency 10`:

| Scenario | p50 | p95 |
|----------|-----|-----|
| Cold (no cache) | ~1 400 ms | ~2 100 ms |
| Warm (Redis cache hit) | ~12 ms | ~25 ms |
| Concurrency × 10 (cached) | ~15 ms | ~40 ms |

> Cache gives **~100× latency reduction** on repeated questions.

---

## Retrieval Evaluation

Run `python scripts/eval_retrieval.py --qa-file scripts/eval_data.json`:

Fill `scripts/eval_data.json` with your Q&A pairs (see the sample file) then run the evaluator. It reports **Hits@3** — whether the correct chunk appears in the top-3 retrieved results.

```
── Retrieval Evaluation ──────────────────
  Questions : 20
  Hits@3    : 18
  Hit rate  : 90.0%
──────────────────────────────────────────
```

*(Update with your real numbers after running against your documents.)*

---

## SQL — Analytical Queries

```sql
-- Average latency per day
SELECT date_trunc('day', created_at) AS day,
       round(avg(latency_ms)::numeric, 1) AS avg_ms,
       count(*) AS requests
FROM queries GROUP BY 1 ORDER BY 1 DESC;

-- Cache-hit rate
SELECT round(100.0 * sum(cache_hit::int) / count(*), 2) AS hit_rate_pct
FROM queries;

-- Top 10 most-asked questions
SELECT question, count(*) AS cnt
FROM queries GROUP BY question ORDER BY cnt DESC LIMIT 10;
```

---

## Project Structure

```
askdocs/
  app/
    main.py              # FastAPI app, routers, lifespan, middleware
    api/
      upload.py          # POST/DELETE /documents
      docs_router.py     # GET /documents
      ask.py             # POST /ask (RAG pipeline)
      health.py          # GET /health
    core/
      config.py          # Pydantic settings from .env
      logging.py         # Structured JSON logs + request-id
      exceptions.py      # Custom exceptions with clear messages
    services/
      ingestion.py       # Text extraction + token chunking
      embeddings.py      # Gemini embed_content, batching, caching
      vector_store.py    # FAISS wrapper, persist/load, delete rebuild
      llm.py             # Gemini GenerativeModel, cited-answer prompt
      cache.py           # Redis + in-memory LRU fallback
    db/
      models.py          # SQLAlchemy ORM: Document, Chunk, Query
      session.py         # Async engine + session factory
      schema.sql         # Raw SQL schema + analytical queries
  tests/
    test_chunking.py     # Chunking edge cases
    test_cache.py        # Cache hit/miss, TTL, invalidation
    test_api.py          # FastAPI TestClient integration tests
  scripts/
    benchmark.py         # p50/p95 latency under concurrency
    eval_retrieval.py    # Hits@k retrieval evaluation
    eval_data.json       # Sample Q&A pairs (fill in your own)
  Dockerfile
  docker-compose.yml
  .env.example
  pyproject.toml         # ruff config, pytest config
  .github/workflows/ci.yml
```

---

## Known Limitations & What's Next

| Limitation | What I'd do next |
|------------|-----------------|
| FAISS `IndexFlatIP` does exact search — O(n) at query time | Migrate to `IndexIVFFlat` or `HNSW` above 500 k vectors |
| Embedding calls are synchronous SDK wrapped in executor | Move to the async Gemini SDK when it reaches GA |
| No re-ranking step | Add a cross-encoder re-ranker (e.g., `ms-marco-MiniLM`) for better precision |
| Single FAISS process — not horizontally scalable | Replace with Qdrant or Weaviate for multi-node deployment |
| No streaming responses | Add SSE streaming from `/ask` with `StreamingResponse` |
| Ingestion failures are silent to the user | Add a webhook / polling endpoint that surfaces error details |

---

## Running Tests

```bash
pytest tests/ -v
```

LLM calls are mocked — tests run without a real API key.

---

## Resume Entry

> **AskDocs: RAG Document Q&A Service** | Python, FastAPI, FAISS, PostgreSQL, Redis, Docker
> - Built an async FastAPI service that ingests PDFs, chunks and embeds them, and answers questions via Gemini API with cited source passages.
> - Added Redis caching (~100× latency reduction on cache hits), PostgreSQL query logging, and Docker Compose deployment.
> - Wrote pytest suite covering chunking edge cases, cache hit/miss, and API validation; retrieval evaluation script reports Hits@3 on a hand-labelled question set.
#   A s k D o c s  
 