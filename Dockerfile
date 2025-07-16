# Production-ready Dockerfile for RAG Chatbot
FROM python:3.11-slim as base

# 시스템 종속성 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 종속성 설치를 위한 레이어 분리
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY src/ ./src/
COPY pyproject.toml .

# 패키지 설치 (editable mode)
RUN pip install -e .

# 데이터 디렉토리 생성
RUN mkdir -p /app/data

# 비루트 사용자 생성
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# 환경 변수 설정
ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=INFO

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 기본 명령어 (웹 서버 시작)
CMD ["rag-chatbot", "serve", "--host", "0.0.0.0", "--port", "8000"]