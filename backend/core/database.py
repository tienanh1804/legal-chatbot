import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Tải biến môi trường
load_dotenv()

# Sử dụng biến môi trường DATABASE_URL hoặc fallback về SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ragapp.db")

# Create SQLAlchemy engine
# Nếu là SQLite, thêm check_same_thread=False
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
