"""User document upload, listing, deletion, summarize, and extract APIs."""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

from auth.auth import get_current_active_user
from core.config import (
    MAX_UPLOAD_BYTES,
    TEXT_CHUNK_OVERLAP,
    TEXT_CHUNK_SIZE,
    USER_DATA_DIR,
)
from core.database import get_db
from core.models import User, UserDocument, UserDocumentChunk
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def _safe_filename(name: str) -> str:
    base = os.path.basename(name).replace("..", "_")
    return re.sub(r"[\r\n\t]", "", base)[:240] or "upload.bin"


def _ensure_user_dir(user_id: int) -> str:
    path = os.path.join(USER_DATA_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


class SummarizeRequest(BaseModel):
    document_ids: Optional[List[int]] = None
    focus: Optional[str] = None


class ExtractRequest(BaseModel):
    document_ids: Optional[List[int]] = None
    instruction: str


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Upload PDF, DOCX, or image; extract text, chunk, embed for user-scoped RAG."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    os.makedirs(USER_DATA_DIR, exist_ok=True)
    user_dir = _ensure_user_dir(current_user.id)
    safe = _safe_filename(file.filename)
    mime = file.content_type or "application/octet-stream"

    row = UserDocument(
        user_id=current_user.id,
        original_filename=safe,
        stored_path="",
        mime_type=mime,
        status="processing",
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    store_name = f"{row.id}_{safe}"
    dest = os.path.join(user_dir, store_name)
    with open(dest, "wb") as f:
        f.write(raw)
    row.stored_path = dest
    db.commit()

    try:
        try:
            from services.document_loaders import extract_text_by_path
            from search.user_document_rag import ingest_file_to_db
        except ImportError:
            from backend.services.document_loaders import extract_text_by_path
            from backend.search.user_document_rag import ingest_file_to_db

        page_count, segments = extract_text_by_path(dest, mime)
        nchunks = ingest_file_to_db(
            db,
            current_user.id,
            row.id,
            page_count,
            segments,
            TEXT_CHUNK_SIZE,
            TEXT_CHUNK_OVERLAP,
        )
    except Exception as e:
        logger.exception("Document processing failed")
        row = db.query(UserDocument).filter(UserDocument.id == row.id).first()
        if row:
            row.status = "failed"
            row.error_message = str(e)[:2000]
            db.commit()
        raise HTTPException(status_code=400, detail=str(e)) from e

    row = db.query(UserDocument).filter(UserDocument.id == row.id).first()
    return {
        "id": row.id,
        "filename": row.original_filename,
        "status": row.status,
        "page_count": row.page_count,
        "chunks_indexed": nchunks,
        "mime_type": row.mime_type,
    }


@router.get("/{document_id}/file")
def get_document_file(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Trả file gốc để xem trước trong trình duyệt (PDF, v.v.)."""
    row = (
        db.query(UserDocument)
        .filter(
            UserDocument.id == document_id,
            UserDocument.user_id == current_user.id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    path = row.stored_path
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not available")
    media = row.mime_type or "application/octet-stream"
    return FileResponse(
        path,
        media_type=media,
        filename=row.original_filename,
        content_disposition_type="inline",
    )


@router.get("")
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    rows = (
        db.query(UserDocument)
        .filter(UserDocument.user_id == current_user.id)
        .order_by(UserDocument.id.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "filename": r.original_filename,
            "status": r.status,
            "page_count": r.page_count,
            "mime_type": r.mime_type,
            "error_message": r.error_message,
        }
        for r in rows
    ]


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        from search.user_document_rag import delete_document_data
    except ImportError:
        from backend.search.user_document_rag import delete_document_data

    row = (
        db.query(UserDocument)
        .filter(
            UserDocument.id == document_id,
            UserDocument.user_id == current_user.id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    path = row.stored_path
    ok = delete_document_data(db, document_id, current_user.id)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            logger.warning("Could not remove file %s", path)
    return {"ok": True}


def _concat_user_document_text(
    db: Session,
    user_id: int,
    document_ids: Optional[List[int]],
) -> str:
    q = (
        db.query(UserDocumentChunk)
        .join(UserDocument, UserDocument.id == UserDocumentChunk.document_id)
        .filter(UserDocument.user_id == user_id)
        .filter(UserDocument.status == "ready")
    )
    if document_ids:
        q = q.filter(UserDocument.id.in_(document_ids))
    chunks = q.order_by(UserDocument.id, UserDocumentChunk.chunk_index).all()
    return "\n\n".join(c.text for c in chunks if c.text)


@router.post("/summarize")
def summarize_documents(
    body: SummarizeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        from services.gemini_text import summarize_text
    except ImportError:
        from backend.services.gemini_text import summarize_text

    text = _concat_user_document_text(db, current_user.id, body.document_ids)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No document text to summarize")
    return {"summary": summarize_text(text, body.focus)}


@router.post("/extract")
def extract_documents(
    body: ExtractRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    if not body.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction is required")
    try:
        from services.gemini_text import extract_key_info
    except ImportError:
        from backend.services.gemini_text import extract_key_info

    text = _concat_user_document_text(db, current_user.id, body.document_ids)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No document text")
    return {"result": extract_key_info(text, body.instruction)}
