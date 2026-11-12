-- AskDocs schema: run once against the askdocs database.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    VARCHAR(512) NOT NULL,
    uploaded_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    status      VARCHAR(32)  NOT NULL DEFAULT 'processing',
    num_chunks  INTEGER      NOT NULL DEFAULT 0,
    file_size_bytes BIGINT   NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_chunks_document_id ON chunks(document_id);

CREATE TABLE IF NOT EXISTS queries (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    latency_ms  FLOAT   NOT NULL,
    cache_hit   BOOLEAN NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_queries_created_at ON queries(created_at);

-- Analytical queries used in benchmarking:

-- Average latency per day:
-- SELECT date_trunc('day', created_at) AS day,
--        round(avg(latency_ms)::numeric, 1) AS avg_ms,
--        count(*) AS requests
-- FROM queries GROUP BY 1 ORDER BY 1 DESC;

-- Cache-hit rate:
-- SELECT round(100.0 * sum(cache_hit::int) / count(*), 2) AS hit_rate_pct
-- FROM queries;

-- Top 10 most-asked questions:
-- SELECT question, count(*) AS cnt
-- FROM queries GROUP BY question ORDER BY cnt DESC LIMIT 10;
