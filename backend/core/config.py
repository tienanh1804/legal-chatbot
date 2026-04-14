import os

from dotenv import load_dotenv

# NOTE:
# - Khi chạy local (không Docker), ta cho phép đọc `.env` để dev tiện cấu hình.
# - Khi chạy trong Docker, env vars đã được docker-compose inject; đọc thêm `.env` trong source
#   có thể vô tình nạp API key cũ và gây lỗi "CONSUMER_SUSPENDED".
if not os.path.exists("/.dockerenv"):
    load_dotenv()

# Document collection configuration
COLLECTION_NAME = "legal_documents"

# Embedding model configuration
EMBEDDING_MODEL = "TienAn1812/legal-embedding-model"
VECTOR_SIZE = 768

# Relational Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ragapp.db")
DB_PATH = os.getenv("DB_PATH", "./data")
CLASS_NAME = os.getenv("CLASS_NAME", COLLECTION_NAME)

# Authentication configuration
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "your-secret-key-for-jwt-please-change-in-production"
)
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 giờ

# Batch processing configuration
BATCH_SIZE = 10

# Input/Output directories
MARKDOWN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "markdown_data"
)

# JSON data directory
JSON_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json_data"
)

# User-uploaded files and per-user indexes (not committed to git by default)
USER_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "user_data"
)
PROCEDURES_TEMPLATE_DIR = os.path.join(JSON_DATA_DIR, "procedures")

# Upload limits
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
USER_RAG_TOP_K = int(os.getenv("USER_RAG_TOP_K", "4"))
TEXT_CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", "900"))
TEXT_CHUNK_OVERLAP = int(os.getenv("TEXT_CHUNK_OVERLAP", "120"))

# Search configuration
TOP_K = 5

# Hybrid Search configuration
USE_HYBRID_SEARCH = True  # Set to False to use only vector search
HYBRID_ALPHA = 0.3  # Default weight for vector search (0-1), adjusted dynamically based on query type
QUERY_EXPANSION = True  # Whether to use query expansion
USE_FAISS = True  # Whether to use FAISS for vector search if available

# Gemini configuration
# Ưu tiên lấy từ ENV để dễ cấu hình khi deploy.
# Mặc định dùng model ổn định, tương thích rộng với API `generateContent`.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = 0.2
GEMINI_MAX_TOKENS = 1024

# Danh sách API key dự phòng
# Lấy từ biến môi trường hoặc sử dụng giá trị mặc định
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY", ""),  # API key chính từ biến môi trường
    os.getenv("GEMINI_API_KEY_1", ""),  # API key dự phòng 1
    os.getenv("GEMINI_API_KEY_2", ""),  # API key dự phòng 2
]

# Lọc bỏ các API key trống
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key]
