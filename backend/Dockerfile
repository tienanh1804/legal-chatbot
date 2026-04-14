# Stage 1: Build dependencies
FROM python:3.9-slim-bullseye as builder

# Cài đặt build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy requirements trước để tận dụng cache
COPY requirements.txt .

# Cài đặt dependencies với các tối ưu
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/\
    && find /install -type d -name "__pycache__" -exec rm -r {} + \
    && find /install -type f -name "*.pyc" -delete

# Stage 2: Runtime
FROM python:3.9-slim-bullseye

# Cài đặt runtime dependencies cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    postgresql-client \
    curl \
    wget \
    antiword \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean autoclean \
    && apt-get autoremove -y

WORKDIR /app

# Copy dependencies từ builder stage
COPY --from=builder /install /usr/local

# Copy source code
COPY . .

# Dọn dẹp các file không cần thiết
RUN find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true \
    && find . -type f -name "*.pyc" -delete \
    && find . -type f -name "*.pyo" -delete \
    && find . -type f -name "*.pyd" -delete \
    && find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true \
    && find . -type d -name "*.dist-info" -exec rm -r {} + 2>/dev/null || true \
    && rm -rf ~/.cache/pip/*

# Thiết lập quyền cho start script
RUN chmod +x start.sh

# Thiết lập biến môi trường Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

CMD ["/app/start.sh"]
