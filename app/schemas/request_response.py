# app/schemas/request_response.py

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Transcribe
# ──────────────────────────────────────────────

class SentimentOut(BaseModel):
    label: str
    score: float


class SegmentOut(BaseModel):
    speaker:   str
    start_s:   float
    end_s:     float
    text:      str
    sentiment: Optional[SentimentOut] = None


class TranscribeResponse(BaseModel):
    call_id:        str
    agent_id:       Optional[str]
    customer_id:    Optional[str]
    language:       str
    duration_s:     float
    raw_transcript: str
    segments:       List[SegmentOut]
    coachable_moments: List[CoachableMomentOut] = []


# ──────────────────────────────────────────────
# Speak
# ──────────────────────────────────────────────

class SpeakRequest(BaseModel):
    text:     str       = Field(..., min_length=1, max_length=5000)
    language: str       = Field(default="en")
    slow:     bool      = Field(default=False)


# ──────────────────────────────────────────────
# Coachable moments
# ──────────────────────────────────────────────

class CoachableMomentOut(BaseModel):
    id:           str
    moment_type:  str
    text:         str
    start_s:      Optional[float]
    end_s:        Optional[float]
    speaker:      Optional[str]
    confidence:   Optional[float]


class ReplayRequest(BaseModel):
    moment_id: str = Field(..., description="ID of the coachable moment to replay")


# ──────────────────────────────────────────────
# Generic
# ──────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str


# Resolve forward reference
TranscribeResponse.model_rebuild()