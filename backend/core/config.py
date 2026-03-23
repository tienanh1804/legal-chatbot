"""Configuration constants for the RAG API.

This file centralizes environment-dependent settings (DB paths, Gemini keys,
retrieval hyperparameters) so other modules can import them safely.
"""

import os
from typing import List

from dotenv import load_dotenv


def _load_dotenv_if_needed() -> None:
    """Load local `.env` when not running inside Docker."""

    # In Docker, env vars are injected; loading a stale local `.env` can cause
    # authentication/config mismatch.
    if not os.path.exists("/.dockerenv"):
        load_dotenv()


_load_dotenv_if_needed()

# Document collection configuration
COLLECTION_NAME: str = "legal_documents"

# Embedding model configuration
EMBEDDING_MODEL: str = "TienAn1812/legal-embedding-model"
VECTOR_SIZE: int = 768

# Relational Database configuration
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ragapp.db")
DB_PATH: str = os.getenv("DB_PATH", "./data")
CLASS_NAME: str = os.getenv("CLASS_NAME", COLLECTION_NAME)

# Authentication configuration
JWT_SECRET_KEY: str = os.getenv(
    "JWT_SECRET_KEY", "your-secret-key-for-jwt-please-change-in-production"
)
JWT_ALGORITHM: str = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

# Batch processing configuration
BATCH_SIZE: int = 10

# Input/Output directories
MARKDOWN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "markdown_data"
)

# JSON data directory
JSON_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json_data"
)

# Search configuration
TOP_K: int = 5

# Hybrid Search configuration
USE_HYBRID_SEARCH: bool = True  # False = vector search only
HYBRID_ALPHA: float = 0.3  # weight for vector search (0-1)
QUERY_EXPANSION: bool = True
USE_FAISS: bool = True

# Gemini configuration
# Ưu tiên lấy từ ENV để dễ cấu hình khi deploy.
# Mặc định dùng model ổn định, tương thích rộng với API `generateContent`.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TEMPERATURE: float = 0.2
GEMINI_MAX_TOKENS: int = 1024

# Danh sách API key dự phòng
# Lấy từ biến môi trường hoặc sử dụng giá trị mặc định
GEMINI_API_KEYS: List[str] = [
    os.getenv("GEMINI_API_KEY", ""),  # API key chính từ biến môi trường
    os.getenv("GEMINI_API_KEY_1", ""),  # API key dự phòng 1
    os.getenv("GEMINI_API_KEY_2", ""),  # API key dự phòng 2
]

# Lọc bỏ các API key trống
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key]
