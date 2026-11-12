"""
FAISS vector store wrapper - persisted to disk so restarts do not lose data.

Consistency guarantee
---------------------
Every chunk id written to FAISS is also the chunk UUID stored in Postgres.
When a document is deleted we remove its chunk ids from both the id-map and
the FAISS index (via selective reconstruction - FAISS IndexFlatIP does not
support in-place removal, so we rebuild from surviving vectors on delete).
"""

import asyncio
import json
from pathlib import Path
from threading import Lock

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import logger

settings = get_settings()

_lock = Lock()  # protects in-process concurrent writes


class VectorStore:
    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._id_map: list[str] = []  # position -> chunk_id
        self._loaded = False

    def _index_path(self) -> Path:
        return Path(settings.faiss_index_path)

    def _id_map_path(self) -> Path:
        return Path(settings.faiss_id_map_path)

    def _ensure_dirs(self) -> None:
        self._index_path().parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """Load index and id-map from disk (called at startup)."""
        self._ensure_dirs()
        if self._index_path().exists() and self._id_map_path().exists():
            self._index = faiss.read_index(str(self._index_path()))
            with self._id_map_path().open() as f:
                self._id_map = json.load(f)
            logger.info("faiss_loaded", vectors=self._index.ntotal)
        else:
            self._index = faiss.IndexFlatIP(settings.embedding_dim)
            self._id_map = []
            logger.info("faiss_new_index", dim=settings.embedding_dim)
        self._loaded = True

    def _save(self) -> None:
        self._ensure_dirs()
        faiss.write_index(self._index, str(self._index_path()))
        with self._id_map_path().open("w") as f:
            json.dump(self._id_map, f)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (vecs / norms).astype(np.float32)

    async def add_vectors(
        self, chunk_ids: list[str], vectors: list[list[float]]
    ) -> None:
        if not self._loaded:
            raise VectorStoreError("Vector store not initialised.")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._add_sync, chunk_ids, vectors)

    def _add_sync(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        mat = self._normalize(np.array(vectors, dtype=np.float32))
        with _lock:
            self._index.add(mat)
            self._id_map.extend(chunk_ids)
            self._save()
        logger.info("faiss_vectors_added", count=len(chunk_ids))

    async def search(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, score) pairs, highest score first."""
        if not self._loaded or self._index.ntotal == 0:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sync, query_vector, top_k)

    def _search_sync(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[str, float]]:
        q = self._normalize(np.array([query_vector], dtype=np.float32))
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)
        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    async def remove_by_chunk_ids(self, chunk_ids: set[str]) -> None:
        """Rebuild the index excluding the given chunk ids."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._rebuild_sync, chunk_ids)

    def _rebuild_sync(self, to_remove: set[str]) -> None:
        with _lock:
            if self._index.ntotal == 0:
                return
            all_vecs = np.zeros(
                (self._index.ntotal, settings.embedding_dim), dtype=np.float32
            )
            for i in range(self._index.ntotal):
                all_vecs[i] = self._index.reconstruct(i)

            new_vecs: list[np.ndarray] = []
            new_ids: list[str] = []
            for i, cid in enumerate(self._id_map):
                if cid not in to_remove:
                    new_vecs.append(all_vecs[i])
                    new_ids.append(cid)

            self._index = faiss.IndexFlatIP(settings.embedding_dim)
            if new_vecs:
                mat = np.vstack(new_vecs).astype(np.float32)
                self._index.add(mat)
            self._id_map = new_ids
            self._save()
        logger.info("faiss_rebuilt", remaining=len(new_ids), removed=len(to_remove))

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0


# Singleton shared across the process
vector_store = VectorStore()
