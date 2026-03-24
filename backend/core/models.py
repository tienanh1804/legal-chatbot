from datetime import datetime

from core.database import Base
from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship with QueryHistory
    queries = relationship("QueryHistory", back_populates="user")

class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    conversation_id = Column(Integer, nullable=True)  # ID của luồng hội thoại
    query_text = Column(Text)
    answer_text = Column(Text)
    sources = Column(Text)  # Store as JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship with User
    user = relationship("User", back_populates="queries")

class HistoryResponse(BaseModel):
    id: int
    user_id: int
    conversation_id: int = None
    query_text: str
    answer_text: str
    sources: str
    created_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}

