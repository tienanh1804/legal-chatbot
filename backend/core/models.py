from datetime import datetime

from core.database import Base
from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, LargeBinary, String, Text
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
    documents = relationship("UserDocument", back_populates="user")
    procedure_sessions = relationship("ProcedureSession", back_populates="user")


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


class UserDocument(Base):
    """Metadata for a file uploaded by a user (PDF/DOCX/image)."""

    __tablename__ = "user_documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    original_filename = Column(String(512))
    stored_path = Column(String(1024))
    mime_type = Column(String(128))
    status = Column(String(32), default="processing")  # processing, ready, failed
    error_message = Column(Text, nullable=True)
    page_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="documents")
    chunks = relationship(
        "UserDocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class UserDocumentChunk(Base):
    """Text chunk + embedding for RAG over user uploads."""

    __tablename__ = "user_document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("user_documents.id"), index=True)
    chunk_index = Column(Integer, default=0)
    page_start = Column(Integer, default=1)
    page_end = Column(Integer, default=1)
    text = Column(Text)
    embedding_blob = Column(LargeBinary, nullable=True)

    document = relationship("UserDocument", back_populates="chunks")


class ProcedureSession(Base):
    """Wizard state for administrative procedure templates."""

    __tablename__ = "procedure_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    template_id = Column(String(128), index=True)
    state_json = Column(Text, default="{}")
    status = Column(String(32), default="active")  # active, completed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="procedure_sessions")


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
