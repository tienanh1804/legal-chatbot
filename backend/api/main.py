import json
import logging
from typing import Dict, Optional

from api import history, users
from auth.auth import get_current_active_user
from core.config import CLASS_NAME, DB_PATH, TOP_K
from core.database import Base, engine, get_db
from core.models import QueryHistory, User
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from search.query_vectordb import format_answer, format_sources, rag_answer_gemini
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API with user authentication and history tracking",
    version="1.0.0",
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:80",
    "http://localhost:3000",
    "http://localhost:5500",  # Live Server default port
    "http://localhost:5501",  # Live Server alternative port
    "http://localhost:5502",  # Live Server alternative port
    "http://localhost:8000",
    "http://localhost:8002",  # Backend port
    "http://localhost:8088",  # Frontend Docker port
    "http://127.0.0.1",
    "http://127.0.0.1:80",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",  # Live Server default port
    "http://127.0.0.1:5501",  # Live Server alternative port
    "http://127.0.0.1:5502",  # Live Server alternative port
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8002",  # Backend port
    "http://127.0.0.1:8088",  # Frontend Docker port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[int] = None


@app.post("/query")
async def handle_query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_active_user),
) -> Dict:
    """Handle incoming query requests."""
    try:
        if not request.query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Received query: {request.query}")

        # Gọi hàm RAG với các tham số cấu hình
        results = rag_answer_gemini(
            query=request.query,
            collection_name=CLASS_NAME,
            top_k=TOP_K,
            db_path=DB_PATH,
            conversation_id=request.conversation_id,  # Truyền conversation_id để lấy lịch sử trò chuyện
        )

        # Ghi log thông tin về API key đã sử dụng (nếu có)
        if "api_key_index" in results:
            logger.info(
                f"Query processed using API key index: {results['api_key_index']}"
            )

        # Ghi log thời gian thực thi
        if "execution_time" in results:
            logger.info(
                f"Query execution time: {results['execution_time']:.2f} seconds"
            )

        response = {
            "response": {
                "answer": format_answer(results),
                "sources": format_sources(results),
                "referenced_doc_ids": results.get(
                    "referenced_doc_ids", []
                ),  # Thêm referenced_doc_ids vào response
            },
            "raw_results": {
                "query": results["query"],
                "answer": results["answer"],
                "sources": results["sources"],
                "model": results["model"],
                "execution_time": results.get("execution_time", 0),
                "referenced_doc_ids": results.get(
                    "referenced_doc_ids", []
                ),  # Thêm referenced_doc_ids vào raw_results
            },
        }

        # Save query to history if user is authenticated
        if current_user:
            # Kiểm tra xem có conversation_id không
            conversation_id = request.conversation_id

            # Nếu không có conversation_id, tạo một cuộc hội thoại mới
            if conversation_id is None:
                # Tìm ID lớn nhất hiện tại để tạo conversation_id mới
                max_id_result = (
                    db.query(QueryHistory).order_by(QueryHistory.id.desc()).first()
                )
                next_id = 1 if max_id_result is None else max_id_result.id + 1
                conversation_id = next_id

            # Lấy nguồn tham khảo đã được định dạng
            formatted_sources = format_sources(results)

            query_history = QueryHistory(
                user_id=current_user.id,
                query_text=request.query,
                answer_text=format_answer(results),
                sources=formatted_sources,
                conversation_id=conversation_id,
            )
            db.add(query_history)
            db.commit()

            # Thêm conversation_id vào response
            response["conversation_id"] = conversation_id

        return response

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/auth/verify")
async def verify_token(current_user: User = Depends(get_current_active_user)):
    """Verify authentication token."""
    return {"status": "valid", "username": current_user.username}


# Include routers
app.include_router(users.router)
app.include_router(history.router)


@app.get("/health", tags=["health"])
def health_check():
    """Health check endpoint for Docker healthcheck"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)  # Sử dụng cổng 8001 thay vì 8000
