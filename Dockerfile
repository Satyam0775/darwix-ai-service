# Dockerfile

FROM python:3.11-slim

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ ./app/
COPY tests/ ./tests/

# Create runtime directories
RUN mkdir -p /tmp/audio_uploads /tmp/tts_output ./hf_cache

# Environment variables
ENV DATABASE_URL=sqlite:///./darwix.db \
    WHISPER_MODEL=base \
    TTS_OUTPUT_DIR=/tmp/tts_output \
    UPLOAD_DIR=tmp/audio_uploads \
    HF_CACHE_DIR=./hf_cache \
    TRANSFORMERS_CACHE=./hf_cache \
    HF_HOME=./hf_cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]


# Build
docker build -t darwix-ai .

# Run
docker run -p 8000:8000 darwix-ai

# Test health
curl http://localhost:8000/health

