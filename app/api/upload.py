"""
Document upload endpoint.

Flow: validate -> save to DB (status=processing) -> return doc_id immediately ->
background task: extract text, chunk, embed, insert chunks, update FAISS, mark ready.

This keeps the HTTP response fast regardless of PDF size.
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.core.logging import logger
from app.db.models import Chunk, Document
from app.db.session import get_db, AsyncSessionLocal
from app.services.cache import cache_service
from app.services.embeddings import embed_texts
from app.services.ingestion import chunk_text, extract_text
from app.services.vector_store import vector_store

settings = get_settings()
router = APIRouter(prefix="/documents", tags=["documents"])


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str


def _validate_file(filename: str, size: int) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise UnsupportedFileTypeError(filename, settings.allowed_extensions)
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if size > max_bytes:
        raise FileTooLargeError(filename, settings.max_file_size_mb)


async def _ingest_document(doc_id: str, filename: str, data: bytes) -> None:
    """Background task: runs the full ingestion pipeline."""
    async with AsyncSessionLocal() as db:
        try:
            # 1. Extract text
            text = await extract_text(filename, data)

            # 2. Chunk
            raw_chunks = chunk_text(text)

            # 3. Embed (with cache)
            texts = [c["text"] for c in raw_chunks]
            vectors = await embed_texts(texts, cache=cache_service)

            # 4. Persist chunks to Postgres
            chunk_objs: list[Chunk] = []
            chunk_ids: list[str] = []
            for raw, vec in zip(raw_chunks, vectors, strict=True):
                cid = str(uuid.uuid4())
                chunk_ids.append(cid)
                chunk_objs.append(
                    Chunk(
                        id=cid,
                        document_id=doc_id,
                        chunk_index=raw["chunk_index"],
                        text=raw["text"],
                        token_count=raw["token_count"],
                    )
                )
            db.add_all(chunk_objs)
            await db.flush()

            # 5. Add to FAISS
            await vector_store.add_vectors(chunk_ids, vectors)

            # 6. Mark document ready
            doc = await db.get(Document, doc_id)
            if doc:
                doc.status = "ready"
                doc.num_chunks = len(chunk_objs)
            await db.commit()
            logger.info("ingestion_complete", doc_id=doc_id, chunks=len(chunk_objs))

        except Exception as exc:
            await db.rollback()
            async with AsyncSessionLocal() as db2:
                doc = await db2.get(Document, doc_id)
                if doc:
                    doc.status = "failed"
                    await db2.commit()
            logger.error("ingestion_failed", doc_id=doc_id, error=str(exc))


@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    data = await file.read()
    filename = file.filename or "upload"

    try:
        _validate_file(filename, len(data))
    except (UnsupportedFileTypeError, FileTooLargeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )

    doc = Document(
        id=str(uuid.uuid4()),
        filename=filename,
        status="processing",
        file_size_bytes=len(data),
    )
    db.add(doc)
    await db.commit()

    background_tasks.add_task(_ingest_document, doc.id, filename, data)

    logger.info("upload_accepted", doc_id=doc.id, filename=filename, size=len(data))
    return UploadResponse(
        document_id=doc.id,
        filename=filename,
        status="processing",
        message="Ingestion started. Poll GET /documents to check status.",
    )


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)) -> None:
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found.",
        )

    # Collect chunk ids before cascade delete
    result = await db.execute(select(Chunk.id).where(Chunk.document_id == doc_id))
    chunk_ids = {row[0] for row in result.fetchall()}

    await db.delete(doc)
    await db.commit()

    # Remove from FAISS
    if chunk_ids:
        await vector_store.remove_by_chunk_ids(chunk_ids)

    # Invalidate cache
    await cache_service.invalidate_by_doc_ids([doc_id])
    logger.info("document_deleted", doc_id=doc_id, chunks_removed=len(chunk_ids))
