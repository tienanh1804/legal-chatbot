"""Administrative procedure templates, wizard sessions, prefill, export."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from auth.auth import get_current_active_user
from core.config import USER_DATA_DIR
from core.database import get_db
from core.models import ProcedureSession, User
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/procedures", tags=["procedures"])


class StartSessionBody(BaseModel):
    template_id: str


class MessageBody(BaseModel):
    message: str


def _state(session: ProcedureSession) -> Dict[str, Any]:
    try:
        from services.procedure_service import merge_state
    except ImportError:
        from backend.services.procedure_service import merge_state

    return merge_state(session.state_json)


def _save_state(db: Session, session: ProcedureSession, state: Dict[str, Any]) -> None:
    session.state_json = json.dumps(state, ensure_ascii=False)
    db.commit()


@router.get("/templates")
def list_templates():
    try:
        from services.procedure_service import list_template_ids, load_template
    except ImportError:
        from backend.services.procedure_service import list_template_ids, load_template

    out: List[Dict[str, Any]] = []
    for tid in list_template_ids():
        try:
            t = load_template(tid)
            out.append(
                {
                    "id": t.get("id", tid),
                    "title": t.get("title", tid),
                    "description": t.get("description", ""),
                    "field_count": len(t.get("fields") or []),
                }
            )
        except OSError:
            continue
    return {"templates": out}


@router.post("/sessions")
def start_session(
    body: StartSessionBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        from services.procedure_service import (
            initial_state,
            load_template,
            next_prompt,
        )
    except ImportError:
        from backend.services.procedure_service import (
            initial_state,
            load_template,
            next_prompt,
        )

    template = load_template(body.template_id)
    state = initial_state(template)
    state, question, done = next_prompt(template, state)
    row = ProcedureSession(
        user_id=current_user.id,
        template_id=template["id"],
        state_json=json.dumps(state, ensure_ascii=False),
        status="active",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {
        "session_id": row.id,
        "template_id": template["id"],
        "question": question,
        "complete": done,
    }


@router.get("/sessions/{session_id}")
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    row = (
        db.query(ProcedureSession)
        .filter(
            ProcedureSession.id == session_id,
            ProcedureSession.user_id == current_user.id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "session_id": row.id,
        "template_id": row.template_id,
        "status": row.status,
        "state": _state(row),
    }


@router.post("/sessions/{session_id}/message")
def session_message(
    session_id: int,
    body: MessageBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        from services.procedure_service import (
            apply_answer,
            load_template,
            next_prompt,
            render_output,
        )
    except ImportError:
        from backend.services.procedure_service import (
            apply_answer,
            load_template,
            next_prompt,
            render_output,
        )

    row = (
        db.query(ProcedureSession)
        .filter(
            ProcedureSession.id == session_id,
            ProcedureSession.user_id == current_user.id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    template = load_template(row.template_id)
    state = _state(row)
    state = apply_answer(template, state, body.message)
    state, question, done = next_prompt(template, state)
    _save_state(db, row, state)
    preview = None
    if done:
        preview = render_output(template, state)
        row.status = "completed"
        db.commit()
    return {
        "question": question,
        "complete": done,
        "collected": state.get("collected"),
        "preview_text": preview,
    }


@router.post("/sessions/{session_id}/prefill-from-documents")
def prefill_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        from search.user_document_rag import aggregate_user_text_for_prefill
        from services.gemini_text import prefill_fields_from_text
        from services.procedure_service import load_template, next_prompt
    except ImportError:
        from backend.search.user_document_rag import aggregate_user_text_for_prefill
        from backend.services.gemini_text import prefill_fields_from_text
        from backend.services.procedure_service import load_template, next_prompt

    row = (
        db.query(ProcedureSession)
        .filter(
            ProcedureSession.id == session_id,
            ProcedureSession.user_id == current_user.id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    template = load_template(row.template_id)
    fields = template.get("fields") or []
    keys = [f.get("key") for f in fields if f.get("key")]
    labels = [f.get("label") or f.get("key") for f in fields if f.get("key")]
    blob = aggregate_user_text_for_prefill(db, current_user.id)
    if not blob.strip():
        raise HTTPException(
            status_code=400, detail="No processed user documents to read from"
        )
    extracted = prefill_fields_from_text(blob, keys, labels)
    state = _state(row)
    collected = dict(state.get("collected") or {})
    collected.update(extracted)
    state["collected"] = collected
    state["step_index"] = 0
    s2, question, done = next_prompt(template, state)
    _save_state(db, row, s2)
    return {
        "extracted": extracted,
        "question": question,
        "complete": done,
        "collected": s2.get("collected"),
    }


@router.get("/sessions/{session_id}/export")
def export_session(
    session_id: int,
    format: str = "docx",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        from services.procedure_export import export_docx_from_template, export_procedure
        from services.procedure_service import load_template, render_output
    except ImportError:
        from backend.services.procedure_export import (
            export_docx_from_template,
            export_procedure,
        )
        from backend.services.procedure_service import load_template, render_output

    row = (
        db.query(ProcedureSession)
        .filter(
            ProcedureSession.id == session_id,
            ProcedureSession.user_id == current_user.id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    template = load_template(row.template_id)
    state = _state(row)
    body = render_output(template, state)
    fmt = (format or "docx").lower()
    out_dir = os.path.join(USER_DATA_DIR, "exports", str(current_user.id))
    os.makedirs(out_dir, exist_ok=True)
    ext = ".docx" if fmt == "docx" else ".pdf"
    out_path = os.path.join(out_dir, f"procedure_{session_id}{ext}")

    # If a DOCX form template is provided, fill it (DOCX only).
    docx_tpl = (template.get("docx_template") or "").strip()
    if fmt == "docx" and docx_tpl:
        try:
            from core.config import PROCEDURES_TEMPLATE_DIR
        except ImportError:
            from backend.core.config import PROCEDURES_TEMPLATE_DIR
        template_path = (
            docx_tpl
            if os.path.isabs(docx_tpl)
            else os.path.join(PROCEDURES_TEMPLATE_DIR, docx_tpl)
        )
        export_docx_from_template(template_path, (state.get("collected") or {}), out_path)
    else:
        export_procedure(body, out_path, fmt)
    media = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if fmt == "docx"
        else "application/pdf"
    )
    return FileResponse(
        out_path,
        filename=f"{template.get('id', 'procedure')}{ext}",
        media_type=media,
    )
