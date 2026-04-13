#!/bin/bash

# Chờ PostgreSQL khởi động
echo "Waiting for PostgreSQL to start..."

# Lấy thông tin kết nối từ DATABASE_URL
if [[ -z "$DATABASE_URL" ]]; then
    # Sử dụng SQLite nếu không có DATABASE_URL
    echo "Using SQLite database"
else
    # Kiểm tra kết nối PostgreSQL
    echo "Checking PostgreSQL connection..."

    # Tách thông tin kết nối từ DATABASE_URL
    # Ví dụ: postgresql://postgres:postgres@db:5432/ragapp
    DB_HOST=$(echo $DATABASE_URL | sed -e 's/^.*@//' -e 's/:.*//')
    DB_PORT=$(echo $DATABASE_URL | sed -e 's/^.*://' -e 's/\/.*//')
    DB_USER=$(echo $DATABASE_URL | sed -e 's/^.*:\/\///' -e 's/:.*$//')
    DB_PASSWORD=$(echo $DATABASE_URL | sed -e 's/^.*:\/\///' -e 's/.*://' -e 's/@.*$//')
    DB_NAME=$(echo $DATABASE_URL | sed -e 's/^.*\///')

    # Chờ kết nối PostgreSQL
    max_retries=30
    retries=0
    until pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER || [ $retries -eq $max_retries ]; do
        echo "Waiting for PostgreSQL to start... ($((retries+1))/$max_retries)"
        sleep 2
        retries=$((retries+1))
    done

    if [ $retries -eq $max_retries ]; then
        echo "Error: Could not connect to PostgreSQL after $max_retries attempts"
        echo "Falling back to SQLite"
        export DATABASE_URL="sqlite:///./ragapp.db"
    else
        echo "PostgreSQL is up and running!"
    fi
fi

# Chạy migration để tạo cơ sở dữ liệu
echo "Running database migrations..."
python scripts/init_db.py

# Ensure cache directory exists and has the right permissions
echo "Ensuring cache directory exists..."
python utils/ensure_cache.py

# Convert safetensors model to pytorch_model.bin
echo "Converting safetensors model to pytorch_model.bin..."
python utils/convert_safetensors_to_pytorch.py

# Move existing cache files to cache directory
echo "Moving cache files to cache directory..."
python utils/move_cache_files.py

# Xây dựng cache cho hybrid search
echo "Building cache for hybrid search..."
python scripts/build_search_resources.py

# Khởi động ứng dụng
echo "Starting the application..."
# UVICORN_RELOAD=1 hoặc true: bật tự tải lại khi sửa code (phù hợp dev).
# Trên Docker + Windows bind mount đôi khi gặp I/O chậm — đặt UVICORN_RELOAD=0 trong .env nếu cần.
if [ "${UVICORN_RELOAD}" = "1" ] || [ "${UVICORN_RELOAD}" = "true" ]; then
  echo "Uvicorn --reload enabled"
  exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app/api --reload-dir /app/auth --reload-dir /app/core --reload-dir /app/search --reload-dir /app/services
else
  exec uvicorn api.main:app --host 0.0.0.0 --port 8000
fi
