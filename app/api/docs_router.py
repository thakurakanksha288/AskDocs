"""Document listing endpoint."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document
from app.db.session import get_db

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    id: str
    filename: str
    uploaded_at: str
    status: str
    num_chunks: int
    file_size_bytes: int

    model_config = {"from_attributes": True}


@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: AsyncSession = Depends(get_db)) -> list[DocumentResponse]:
    result = await db.execute(
        select(Document).order_by(Document.uploaded_at.desc())
    )
    docs = result.scalars().all()
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            uploaded_at=doc.uploaded_at.isoformat(),
            status=doc.status,
            num_chunks=doc.num_chunks,
            file_size_bytes=doc.file_size_bytes,
        )
        for doc in docs
    ]
