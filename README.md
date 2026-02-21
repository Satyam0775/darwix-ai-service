# Darwix AI — Sales Call Intelligence Microservice

A production-grade Python microservice that processes sales-call audio snippets,
generates transcripts, detects coachable moments, performs sentiment analysis,
and provides text-to-speech playback for agent training.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| Speech-to-Text | OpenAI Whisper (base model) |
| Sentiment Analysis | DistilBERT (HuggingFace) |
| Text-to-Speech | gTTS (Google TTS) |
| Speaker Diarization | Whisper-based segmentation (HF diarization optional) |
| Database | SQLite (via SQLAlchemy ORM) |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Logging | Loguru |
| Fault Tolerance | Tenacity retry + graceful fallbacks |

---

## Project Structure


darwix-ai-service/
├── .github/
│ └── workflows/
│ └── ci.yml
├── app/
│ ├── api/
│ ├── db/
│ ├── schemas/
│ ├── services/
│ └── main.py
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md


---

## Local Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/Satyam0775/darwix-ai-service.git
cd darwix-ai-service
2. Create virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Start the server
uvicorn app.main:app --reload

Visit:

http://127.0.0.1:8000/docs
Docker Setup
docker build -t darwix-ai .
docker run -p 8000:8000 darwix-ai
API Endpoints
POST /transcribe
curl -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@speech.mp3" \
  -F "call_id=call_001" \
  -F "agent_id=agent_01" \
  -F "customer_id=customer_01"
POST /speak
curl -X POST http://127.0.0.1:8000/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "I am not sure about the price."}' \
  --output speech_out.mp3
POST /replay
curl -X POST http://127.0.0.1:8000/replay \
  -H "Content-Type: application/json" \
  -d '{"call_id": "call_001"}' \
  --output replay.mp3
GET /replay/list/{call_id}
curl http://127.0.0.1:8000/replay/list/call_001
GET /health
curl http://127.0.0.1:8000/health
Running Tests
pytest tests/ -v
Architectural Decisions

Modular separation of STT, TTS, sentiment, coaching, DB, and API layers
Whisper base model for cost-free local inference
DistilBERT sentiment with retry mechanism
Rule-based coachable moment detection for explainability
SQLite for MVP; easily switchable to PostgreSQL
Stateless design for horizontal scaling

Scalability
Horizontal scaling via multiple FastAPI instances
Upgrade path to Celery + Redis for async processing
Replace SQLite with PostgreSQL in production
Dedicated GPU worker for Whisper in high-load systems

Fault Tolerance
Tenacity retry on model inference
Graceful fallback to NEUTRAL sentiment
Fallback speaker labeling if diarization unavailable
DB rollback on error
Structured logging with Loguru
