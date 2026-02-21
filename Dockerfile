FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /tmp/audio_uploads /tmp/tts_output ./hf_cache

ENV DATABASE_URL=sqlite:///./darwix.db \
    WHISPER_MODEL=base \
    TTS_OUTPUT_DIR=/tmp/tts_output \
    UPLOAD_DIR=tmp/audio_uploads \
    HF_CACHE_DIR=./hf_cache \
    TRANSFORMERS_CACHE=./hf_cache \
    HF_HOME=./hf_cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
