import json
from datetime import datetime
from typing import Dict, List, Optional

from auth.auth import get_current_active_user
from core import models
from core.database import get_db
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

router = APIRouter(
    prefix="/history",
    tags=["history"],
    responses={404: {"description": "Not found"}},
)


class QueryHistoryCreate(BaseModel):
    query_text: str
    answer_text: str
    sources: str
    conversation_id: Optional[int] = None


class QueryHistoryResponse(BaseModel):
    id: int
    query_text: str
    answer_text: str
    sources: str
    created_at: datetime
    conversation_id: Optional[int] = None

    class Config:
        orm_mode = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}


@router.post("/", response_model=QueryHistoryResponse)
def create_query_history(
    history: QueryHistoryCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Save a query to history."""
    # Kiểm tra xem có conversation_id không
    conversation_id = history.conversation_id

    # Nếu không có conversation_id, tạo một cuộc hội thoại mới
    if conversation_id is None:
        # Tìm ID lớn nhất hiện tại để tạo conversation_id mới
        max_id_result = (
            db.query(models.QueryHistory)
            .order_by(models.QueryHistory.id.desc())
            .first()
        )
        next_id = 1 if max_id_result is None else max_id_result.id + 1
        conversation_id = next_id

    db_history = models.QueryHistory(
        user_id=current_user.id,
        query_text=history.query_text,
        answer_text=history.answer_text,
        sources=history.sources,
        conversation_id=conversation_id,
    )
    db.add(db_history)
    db.commit()
    db.refresh(db_history)
    return db_history


@router.get("/", response_model=List[QueryHistoryResponse])
def get_user_history(
    skip: int = 0,
    limit: int = 100,
    group_by_conversation: bool = True,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Get user's query history.

    If group_by_conversation is True, returns only the first message of each conversation.
    """
    # Truy vấn cơ bản
    query = db.query(models.QueryHistory).filter(
        models.QueryHistory.user_id == current_user.id
    )

    if group_by_conversation and limit > 0:
        # SQLite không hỗ trợ DISTINCT ON nên chúng ta cần sử dụng cách tiếp cận khác
        # Lấy danh sách các conversation_id duy nhất
        conversation_ids = (
            db.query(models.QueryHistory.conversation_id)
            .filter(models.QueryHistory.user_id == current_user.id)
            .distinct()
            .all()
        )

        # Chuyển kết quả thành danh sách các conversation_id
        conversation_ids = [cid[0] for cid in conversation_ids if cid[0] is not None]

        # Danh sách lưu các ID của tin nhắn đầu tiên trong mỗi luồng
        first_message_ids = []

        # Với mỗi conversation_id, lấy tin nhắn đầu tiên
        for cid in conversation_ids:
            first_message = (
                db.query(models.QueryHistory)
                .filter(
                    models.QueryHistory.user_id == current_user.id,
                    models.QueryHistory.conversation_id == cid,
                )
                .order_by(models.QueryHistory.created_at.asc())
                .first()
            )
            if first_message:
                first_message_ids.append(first_message.id)

        # Lấy các tin nhắn đầu tiên của mỗi luồng
        query = (
            db.query(models.QueryHistory)
            .filter(models.QueryHistory.id.in_(first_message_ids))
            .order_by(models.QueryHistory.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
    else:
        # Nếu không nhóm, lấy tất cả các tin nhắn
        query = (
            query.order_by(models.QueryHistory.created_at.desc())
            .offset(skip)
            .limit(limit)
        )

    history = query.all()
    return history


@router.get("/{history_id}", response_model=QueryHistoryResponse)
def get_history_item(
    history_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Get a specific history item."""
    history_item = (
        db.query(models.QueryHistory)
        .filter(
            models.QueryHistory.id == history_id,
            models.QueryHistory.user_id == current_user.id,
        )
        .first()
    )
    if history_item is None:
        raise HTTPException(status_code=404, detail="History item not found")
    return history_item


@router.get(
    "/conversation/{conversation_id}", response_model=List[QueryHistoryResponse]
)
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Get all messages in a conversation."""
    conversation = (
        db.query(models.QueryHistory)
        .filter(
            models.QueryHistory.conversation_id == conversation_id,
            models.QueryHistory.user_id == current_user.id,
        )
        .order_by(models.QueryHistory.created_at.asc())
        .all()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.delete("/{history_id}")
def delete_history_item(
    history_id: int,
    delete_conversation: bool = False,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Delete a history item or an entire conversation.

    If delete_conversation is True, deletes all messages in the conversation.
    """
    # Lấy tin nhắn cần xóa
    history_item = (
        db.query(models.QueryHistory)
        .filter(
            models.QueryHistory.id == history_id,
            models.QueryHistory.user_id == current_user.id,
        )
        .first()
    )

    if history_item is None:
        raise HTTPException(status_code=404, detail="History item not found")

    # Nếu xóa cả luồng hội thoại
    if delete_conversation and history_item.conversation_id is not None:
        deleted_count = (
            db.query(models.QueryHistory)
            .filter(
                models.QueryHistory.conversation_id == history_item.conversation_id,
                models.QueryHistory.user_id == current_user.id,
            )
            .delete()
        )
        db.commit()
        return {"detail": f"{deleted_count} history items deleted"}
    else:
        # Xóa chỉ một tin nhắn
        db.delete(history_item)
        db.commit()
        return {"detail": "History item deleted"}


@router.delete("/")
def delete_all_history(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Delete all history items for the current user."""
    deleted_count = (
        db.query(models.QueryHistory)
        .filter(models.QueryHistory.user_id == current_user.id)
        .delete()
    )
    db.commit()
    return {"detail": f"{deleted_count} history items deleted"}
