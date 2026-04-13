"""Per-user document chunking, embedding, and similarity search."""

from __future__ import annotations

import logging
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def chunk_text_segments(
    segments: List[Tuple[int, int, str]],
    chunk_size: int,
    overlap: int,
) -> List[Tuple[int, int, str]]:
    """
    Flatten page segments into overlapping chunks with page ranges.

    Each output item: (page_start, page_end, chunk_text).
    """
    out: List[Tuple[int, int, str]] = []
    for ps, pe, text in segments:
        if not text.strip():
            continue
        # Split long page text into windows
        t = text.strip()
        if len(t) <= chunk_size:
            out.append((ps, pe, t))
            continue
        start = 0
        while start < len(t):
            end = min(start + chunk_size, len(t))
            piece = t[start:end]
            out.append((ps, pe, piece.strip()))
            if end >= len(t):
                break
            start = max(0, end - overlap)
    return out


def _get_embedding_fn():
    try:
        from search.query_vectordb import get_embedding
    except ImportError:
        from backend.search.query_vectordb import get_embedding
    return get_embedding


def _cosine(a: List[float], b: List[float]) -> float:
    import math

    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return s / (na * nb)


def search_user_chunks(
    db: Session,
    user_id: int,
    query: str,
    top_k: int,
    document_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Vector search over the current user's chunks (cosine on stored embeddings).

    If document_ids is set, restrict to those user_documents.id values.
    """
    try:
        from core.models import UserDocument, UserDocumentChunk
    except ImportError:
        from backend.core.models import UserDocument, UserDocumentChunk

    q = (
        db.query(UserDocumentChunk)
        .join(UserDocument, UserDocument.id == UserDocumentChunk.document_id)
        .filter(UserDocument.user_id == user_id)
        .filter(UserDocument.status == "ready")
    )
    if document_ids:
        q = q.filter(UserDocument.id.in_(document_ids))

    chunks = q.all()
    if not chunks:
        return []

    get_embedding = _get_embedding_fn()
    qvec = get_embedding(query)

    scored: List[Tuple[float, Any]] = []
    for ch in chunks:
        if not ch.embedding_blob:
            continue
        try:
            ev = pickle.loads(ch.embedding_blob)
        except Exception:
            continue
        score = _cosine(qvec, ev)
        scored.append((score, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    results: List[Dict[str, Any]] = []
    for score, ch in scored[:top_k]:
        doc = db.query(UserDocument).filter(UserDocument.id == ch.document_id).first()
        if not doc:
            continue
        results.append(
            {
                "score": float(score),
                "chunk_id": ch.id,
                "chunk_index": ch.chunk_index,
                "document_id": doc.id,
                "filename": doc.original_filename,
                "page_start": ch.page_start,
                "page_end": ch.page_end,
                "text": ch.text,
                "ref": f"U{doc.id}-{ch.chunk_index}",
            }
        )
    return results


_VAGUE_USER_DOC = re.compile(
    r"t[oó]m\s*t[ắa]t|summar|gi[ảa]i\s*th[íi]ch|l[àa]m\s*r[õo]|"
    r"t[àa]i\s*li[ệe]u|n[ộo]i\s*dung|\bfile\b|đ[ọo]an|chi\s*ti[ếe]t|"
    r"gi[úu]p\s*t[ôơ]i|h[ãa]y\s*t[óo]m|l[àa]m\s*ro|r[õo]\s*h[ơo]n",
    re.IGNORECASE,
)


def is_vague_user_document_query(query: str) -> bool:
    """Short/meta questions (tóm tắt, giải thích…) — semantic search alone is weak."""
    q = (query or "").strip()
    if len(q) > 180:
        return False
    return bool(_VAGUE_USER_DOC.search(q))


def get_recent_user_chunks(
    db: Session,
    user_id: int,
    limit: int = 24,
    document_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Latest uploaded docs first, chunks in order — fallback when query is vague."""
    try:
        from core.models import UserDocument, UserDocumentChunk
    except ImportError:
        from backend.core.models import UserDocument, UserDocumentChunk

    q = (
        db.query(UserDocumentChunk, UserDocument)
        .join(UserDocument, UserDocument.id == UserDocumentChunk.document_id)
        .filter(UserDocument.user_id == user_id)
        .filter(UserDocument.status == "ready")
    )
    if document_ids:
        q = q.filter(UserDocument.id.in_(document_ids))
    rows = (
        q.order_by(UserDocument.id.desc(), UserDocumentChunk.chunk_index.asc())
        .limit(limit)
        .all()
    )
    out: List[Dict[str, Any]] = []
    for ch, doc in rows:
        out.append(
            {
                "score": 1.0,
                "chunk_id": ch.id,
                "chunk_index": ch.chunk_index,
                "document_id": doc.id,
                "filename": doc.original_filename,
                "page_start": ch.page_start,
                "page_end": ch.page_end,
                "text": ch.text,
                "ref": f"U{doc.id}-{ch.chunk_index}",
            }
        )
    return out


def merge_semantic_and_recent_user_chunks(
    db: Session,
    user_id: int,
    query: str,
    top_k: int = 4,
    max_merged: int = 14,
    document_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Combine vector search with recent chunks when the query is vague or matches are weak
    (e.g. 'tóm tắt giúp tôi' does not align well with embedding of legal PDF text).
    """
    semantic = search_user_chunks(
        db, user_id, query, top_k=top_k, document_ids=document_ids
    )
    top_score = semantic[0]["score"] if semantic else 0.0
    need_more = (
        not semantic
        or is_vague_user_document_query(query)
        or len(semantic) < 2
        or top_score < 0.14
    )
    if not need_more:
        return semantic[:max_merged]

    recent = get_recent_user_chunks(
        db, user_id, limit=max(28, top_k * 5), document_ids=document_ids
    )
    seen = {h["ref"] for h in semantic}
    merged: List[Dict[str, Any]] = list(semantic)
    for h in recent:
        if h["ref"] in seen:
            continue
        merged.append(h)
        seen.add(h["ref"])
        if len(merged) >= max_merged:
            break
    return merged[:max_merged]


def get_latest_ready_document_id(db: Session, user_id: int) -> Optional[int]:
    """Newest successfully indexed user upload, for scoping Q&A to one file."""
    try:
        from core.models import UserDocument
    except ImportError:
        from backend.core.models import UserDocument

    row = (
        db.query(UserDocument)
        .filter(UserDocument.user_id == user_id, UserDocument.status == "ready")
        .order_by(UserDocument.id.desc())
        .first()
    )
    return int(row.id) if row else None


def delete_document_data(db: Session, document_id: int, user_id: int) -> bool:
    try:
        from core.models import UserDocument, UserDocumentChunk
    except ImportError:
        from backend.core.models import UserDocument, UserDocumentChunk

    doc = (
        db.query(UserDocument)
        .filter(UserDocument.id == document_id, UserDocument.user_id == user_id)
        .first()
    )
    if not doc:
        return False
    db.query(UserDocumentChunk).filter(
        UserDocumentChunk.document_id == document_id
    ).delete()
    db.delete(doc)
    db.commit()
    return True


def ingest_file_to_db(
    db: Session,
    user_id: int,
    document_row_id: int,
    page_count: int,
    chunk_items: List[Tuple[int, int, str]],
    chunk_size: int,
    overlap: int,
) -> int:
    """Create chunks + embeddings for a user document. Returns number of chunks."""
    try:
        from core.models import UserDocument, UserDocumentChunk
    except ImportError:
        from backend.core.models import UserDocument, UserDocumentChunk

    doc = (
        db.query(UserDocument)
        .filter(UserDocument.id == document_row_id, UserDocument.user_id == user_id)
        .first()
    )
    if not doc:
        raise ValueError("Document not found")

    flat = chunk_text_segments(chunk_items, chunk_size, overlap)
    get_embedding = _get_embedding_fn()

    db.query(UserDocumentChunk).filter(
        UserDocumentChunk.document_id == document_row_id
    ).delete()

    n = 0
    for idx, (ps, pe, txt) in enumerate(flat):
        if not txt.strip():
            continue
        emb = get_embedding(txt[:8000])
        blob = pickle.dumps(emb, protocol=pickle.HIGHEST_PROTOCOL)
        row = UserDocumentChunk(
            document_id=document_row_id,
            chunk_index=idx,
            page_start=ps,
            page_end=pe,
            text=txt[:65000],
            embedding_blob=blob,
        )
        db.add(row)
        n += 1

    doc.page_count = page_count
    doc.status = "ready"
    doc.error_message = None
    db.commit()
    return n


def aggregate_user_text_for_prefill(
    db: Session, user_id: int, max_chars: int = 12000
) -> str:
    """Concatenate recent user document text for procedure prefill / extraction."""
    try:
        from core.models import UserDocument, UserDocumentChunk
    except ImportError:
        from backend.core.models import UserDocument, UserDocumentChunk

    docs = (
        db.query(UserDocument)
        .filter(UserDocument.user_id == user_id, UserDocument.status == "ready")
        .order_by(UserDocument.id.desc())
        .limit(10)
        .all()
    )
    parts: List[str] = []
    total = 0
    for d in docs:
        chunks = (
            db.query(UserDocumentChunk)
            .filter(UserDocumentChunk.document_id == d.id)
            .order_by(UserDocumentChunk.chunk_index)
            .all()
        )
        for c in chunks:
            block = f"--- {d.original_filename} (trang {c.page_start}-{c.page_end}) ---\n{c.text}\n"
            if total + len(block) > max_chars:
                remain = max_chars - total
                if remain > 100:
                    parts.append(block[:remain])
                return "\n".join(parts)
            parts.append(block)
            total += len(block)
    return "\n".join(parts)
