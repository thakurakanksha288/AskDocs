# AskDocs

**Ask questions about your documents and get answers with the source passages cited.**

AskDocs is an async RAG (retrieval-augmented generation) service. Upload a PDF and it extracts the text, splits it into chunks, embeds them and indexes them in FAISS. When you ask a question, it retrieves the most relevant passages and sends them to an LLM that answers only from that context. If the document doesn't contain the answer, the service says so instead of guessing.

![Demo](docs/demo.gif)
<!-- TODO: record a 20-30 second GIF of upload -> ask -> cited answer and save it as docs/demo.gif -->

**Stack:** Python · FastAPI · FAISS · PostgreSQL · Redis · Docker · pytest

---

## Features

- **Async API.** FastAPI endpoints with `async`/`await`, an async DB driver and non-blocking LLM calls.
- **Background ingestion.** Large uploads return immediately with a document id; processing status can be polled.
- **RAG pipeline.** Configurable chunking, batched embeddings, FAISS vector search and a grounded prompt that returns cited sources.
- **PostgreSQL storage.** Documents, chunks and query logs in a relational schema with indexes.
- **Redis caching.** Repeated questions are answered from cache, with an in-memory fallback if Redis is unavailable.
- **Observability.** Structured JSON logs with a request id and per-stage timings (embed, retrieve, generate).
- **Tested.** pytest suite with the LLM mocked, so tests run without an API key.
- **One-command setup.** `docker compose up` starts the API, Postgres and Redis.

> Only keep the lines above that are implemented in your repo. Delete the rest until they are built.

---

## Architecture

```
Client (Swagger UI / simple web page)
              |
        FastAPI (async)
              |
   +----------+-----------------------+
   |                                  |
Ingestion pipeline               Query pipeline
extract -> chunk -> embed        cache check -> embed question
   |                             -> retrieve top-k -> LLM -> cite sources
   |                                  |
   +--> PostgreSQL (documents,        +--> Redis (answer + embedding cache)
   |    chunks, query logs)           +--> LLM API
   +--> FAISS (vector index)
```

<!-- TODO: replace with a drawn diagram at docs/architecture.png -->

### Request flow for `/ask`

1. Hash the question plus document ids and check Redis. A hit returns immediately.
2. Embed the question.
3. Search FAISS for the top-k chunks.
4. Load the chunk text from PostgreSQL.
5. Build a prompt that tells the model to answer only from the provided context, cite chunk ids and say "I don't know" when the context lacks the answer.
6. Call the LLM, store the answer in Redis and log the query in PostgreSQL.

---

## Quick start

### Requirements
- Docker and Docker Compose
- An API key for your LLM provider (Gemini or OpenAI)

### Run

```bash
git clone https://github.com/<your-username>/askdocs-rag.git
cd askdocs-rag
cp .env.example .env        # add your API key
docker compose up --build
```

Open **http://localhost:8000/docs** for the interactive API docs.

### Try it

```bash
# Upload a document
curl -F "file=@sample.pdf" http://localhost:8000/documents

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main conclusion?", "top_k": 4}'
```

Example response:

```json
{
  "answer": "...",
  "sources": [
    {"chunk_id": 12, "document": "sample.pdf", "text": "..."}
  ],
  "cached": false,
  "latency_ms": 0
}
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/documents` | Upload a PDF or text file; returns a document id |
| GET | `/documents` | List uploaded documents and their processing status |
| DELETE | `/documents/{id}` | Delete a document, its chunks and its vectors |
| POST | `/ask` | Ask a question; returns an answer with cited sources |
| GET | `/health` | Checks PostgreSQL, Redis and the vector index |

---

## Configuration

Set these in `.env`:

| Variable | Description | Default |
|---|---|---|
| `LLM_API_KEY` | API key for the LLM provider | required |
| `LLM_MODEL` | Model used for answers | your choice |
| `EMBEDDING_MODEL` | Model used for embeddings | your choice |
| `CHUNK_SIZE` | Tokens per chunk | 500 |
| `CHUNK_OVERLAP` | Tokens shared between chunks | 50 |
| `TOP_K` | Chunks retrieved per question | 4 |
| `CACHE_TTL_SECONDS` | Answer cache lifetime | 3600 |
| `DATABASE_URL` | PostgreSQL connection string | set in compose |
| `REDIS_URL` | Redis connection string | set in compose |

---

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Chunk size | 500 tokens, 50 overlap | Small enough for precise retrieval, large enough to keep context; overlap avoids cutting an answer in half. |
| Vector index | FAISS | Fast local similarity search with no extra service to run. |
| Metadata store | PostgreSQL | Chunk text, document status and query logs need relational queries and indexes. |
| Caching | Redis, keyed by hash of question + document ids | Repeated questions skip embedding and the LLM call. |
| Concurrency | `asyncio` for I/O, thread pool for PDF parsing | LLM, DB and cache calls are I/O-bound; PDF parsing blocks the event loop. |

<!-- TODO: edit this table to match what you actually chose and why. -->

### Database schema

```sql
documents(id, filename, status, num_chunks, uploaded_at)
chunks(id, document_id, chunk_index, text, token_count)
queries(id, question, answer, latency_ms, cache_hit, created_at)
```

Example analytics queries:

```sql
-- Cache hit rate
SELECT AVG(cache_hit::int) AS hit_rate FROM queries;

-- Average latency per day
SELECT date_trunc('day', created_at) AS day, AVG(latency_ms) AS avg_ms
FROM queries GROUP BY 1 ORDER BY 1;
```

---

## Benchmarks

Measured with `scripts/benchmark.py` on `<your machine>`, `<N>` requests:

| Scenario | p50 latency | p95 latency |
|---|---|---|
| No cache | TODO ms | TODO ms |
| With Redis cache | TODO ms | TODO ms |

**Bottleneck found by profiling:** TODO (for example, embedding calls; fixed by batching).

<!-- Fill this table only with numbers you actually measured. Delete the section if you haven't benchmarked yet. -->

## Retrieval evaluation

`scripts/eval_retrieval.py` runs a hand-labelled set of `<N>` questions on `<document>` and checks whether the correct chunk appears in the top 3 results.

| Metric | Result |
|---|---|
| Hit rate @ 3 | TODO % |

<!-- Delete this section if you haven't run an evaluation. -->

---

## Project structure

```
askdocs-rag/
  app/
    main.py              # FastAPI app and startup
    api/                 # documents.py, ask.py, health.py
    core/                # config.py, logging.py
    services/
      ingestion.py       # text extraction and chunking
      embeddings.py      # batched embedding calls
      vector_store.py    # FAISS wrapper
      llm.py             # prompt building and LLM client
      cache.py           # Redis with in-memory fallback
    db/                  # models.py, session.py
  tests/
  scripts/               # benchmark.py, eval_retrieval.py
  docker-compose.yml
  Dockerfile
  README.md
```

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Tests cover chunking edge cases, retrieval of a known passage, API endpoints and cache hit/miss behavior. The LLM client is mocked.

---

## Limitations and next steps

- Only text-based PDFs are supported; scanned documents need OCR.
- Retrieval uses vector similarity only; adding keyword search (BM25) or a re-ranker should help on exact-match queries.
- FAISS runs in a single process; a managed vector database would be needed to scale to millions of chunks.
- No authentication or per-user document isolation yet.

<!-- Keep this list honest and specific. It shows you understand the system's limits. -->

---

## License

MIT