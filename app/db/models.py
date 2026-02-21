# app/db/models.py

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Float, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class CallRecord(Base):
    """Stores metadata and full transcript for a single audio call."""
    __tablename__ = "call_records"

    id          = Column(String, primary_key=True, default=_uuid)
    call_id     = Column(String, nullable=False, index=True)
    agent_id    = Column(String, nullable=True)
    customer_id = Column(String, nullable=True)
    audio_path  = Column(String, nullable=True)          # path to stored audio
    raw_transcript = Column(Text, nullable=True)         # full plain text
    language    = Column(String, default="en")
    duration_s  = Column(Float, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)

    segments        = relationship("TranscriptSegment", back_populates="call",
                                   cascade="all, delete-orphan")
    coachable_moments = relationship("CoachableMoment", back_populates="call",
                                     cascade="all, delete-orphan")


class TranscriptSegment(Base):
    """Individual diarized + sentiment-tagged utterance."""
    __tablename__ = "transcript_segments"

    id          = Column(String, primary_key=True, default=_uuid)
    call_id_fk  = Column(String, ForeignKey("call_records.id"), nullable=False)
    speaker     = Column(String, nullable=False)          # e.g. "SPEAKER_00"
    start_s     = Column(Float, nullable=False)
    end_s       = Column(Float, nullable=False)
    text        = Column(Text, nullable=False)
    sentiment   = Column(String, nullable=True)           # POSITIVE/NEGATIVE/NEUTRAL
    sentiment_score = Column(Float, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)

    call = relationship("CallRecord", back_populates="segments")


class CoachableMoment(Base):
    """A detected coaching opportunity inside a call."""
    __tablename__ = "coachable_moments"

    id          = Column(String, primary_key=True, default=_uuid)
    call_id_fk  = Column(String, ForeignKey("call_records.id"), nullable=False)
    moment_type = Column(String, nullable=False)          # e.g. "objection", "buying_signal"
    text        = Column(Text, nullable=False)
    start_s     = Column(Float, nullable=True)
    end_s       = Column(Float, nullable=True)
    speaker     = Column(String, nullable=True)
    confidence  = Column(Float, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)

    call = relationship("CallRecord", back_populates="coachable_moments")