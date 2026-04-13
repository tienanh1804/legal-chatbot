import logging
from typing import Dict, Optional

from api import documents, history, procedures, users
from auth.auth import get_current_active_user, get_current_active_user_optional
from core.config import CLASS_NAME, DB_PATH, TOP_K
from core.database import Base, engine, get_db
from core.models import QueryHistory, User
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from search.query_vectordb import format_answer, format_sources, rag_answer_gemini
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Base.metadata.create_all(bind=engine)
app = FastAPI(title="RAG API", version="1.0.0")
origins = [
    "http://localhost",
    "http://localhost:80",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://localhost:5501",
    "http://localhost:5502",
    "http://localhost:8000",
    "http://localhost:8002",
    "http://localhost:8088",
    "http://127.0.0.1",
    "http://127.0.0.1:80",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5501",
    "http://127.0.0.1:5502",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8088",
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[int] = None
    include_user_documents: bool = True
    # True: chỉ dùng đoạn trích file đã tải, không truy vấn kho VBQPPL (tránh lẫn nội dung)
    user_documents_only: bool = False
    user_document_id: Optional[int] = None  # file vừa gắn; None thì dùng tài liệu mới nhất

@app.post("/query")
async def handle_query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_active_user_optional),
) -> Dict:
    def _get_or_create_conversation_id(conversation_id: Optional[int]) -> int:
        if conversation_id is not None:
            return conversation_id
        max_id_result = db.query(QueryHistory).order_by(QueryHistory.id.desc()).first()
        return 1 if max_id_result is None else max_id_result.id + 1
    def _save_history(conversation_id: int, answer_text: str, sources_text: str) -> None:
        db.add(QueryHistory(user_id=current_user.id, query_text=request.query, answer_text=answer_text, sources=sources_text, conversation_id=conversation_id))
        db.commit()
    try:
        if not request.query.strip():
            raise ValueError("Query cannot be empty")
        if request.user_documents_only and not current_user:
            raise ValueError(
                "Cần đăng nhập để trả lời chỉ trong phạm vi tài liệu đã tải lên."
            )
        results = rag_answer_gemini(
            query=request.query,
            collection_name=CLASS_NAME,
            top_k=TOP_K,
            db_path=DB_PATH,
            conversation_id=request.conversation_id,
            db=db,
            user_id=current_user.id if current_user else None,
            include_user_documents=request.include_user_documents,
            user_documents_only=request.user_documents_only,
            user_document_id=request.user_document_id,
        )
        response = {"response": {"answer": format_answer(results), "sources": format_sources(results), "referenced_doc_ids": results.get("referenced_doc_ids", []), "referenced_user_refs": results.get("referenced_user_refs", []), "user_sources": results.get("user_sources", []), "answer_mode": results.get("answer_mode")}, "raw_results": {"query": results["query"], "answer": results["answer"], "sources": results["sources"], "user_sources": results.get("user_sources", []), "model": results["model"], "execution_time": results.get("execution_time", 0), "referenced_doc_ids": results.get("referenced_doc_ids", []), "referenced_user_refs": results.get("referenced_user_refs", []), "answer_mode": results.get("answer_mode")}}
        if current_user:
            cid = _get_or_create_conversation_id(request.conversation_id)
            _save_history(cid, format_answer(results), format_sources(results))
            response["conversation_id"] = cid
        return response
    except ValueError as ve:
        if current_user:
            cid = _get_or_create_conversation_id(request.conversation_id)
            _save_history(cid, str(ve), "")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("%s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/auth/verify")
async def verify_token(current_user: User = Depends(get_current_active_user)):
    return {"status": "valid", "username": current_user.username}

app.include_router(users.router)
app.include_router(history.router)
app.include_router(documents.router)
app.include_router(procedures.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


