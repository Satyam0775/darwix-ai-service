# app/main.py

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.db.database import init_db
from app.api import transcribe, speak, replay


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Darwix AI Service starting up...")
    init_db()
    logger.info("✅ Database ready.")
    yield
    logger.info("👋 Darwix AI Service shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Darwix AI – Sales Call Intelligence Service",
    description=(
        "Microservice for processing sales-call audio:\n"
        "- **POST /transcribe** — STT + diarization + sentiment + coachable moments\n"
        "- **POST /speak** — Text-to-Speech synthesis\n"
        "- **POST /replay** — Replay a coachable moment via TTS\n"
        "- **GET  /replay/list/{call_id}** — List all coachable moments for a call"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(transcribe.router)
app.include_router(speak.router)
app.include_router(replay.router)


# ── Health & Root ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="Service health check")
async def health_check():
    return {
        "status": "ok",
        "service": "darwix-ai",
        "version": "1.0.0",
    }


@app.get("/", tags=["Health"], include_in_schema=False)
async def root():
    return {
        "message": "Darwix AI Service is running. Visit /docs for API documentation."
    }