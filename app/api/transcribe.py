# app/api/transcribe.py

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import CallRecord, CoachableMoment, TranscriptSegment
from app.schemas.request_response import (
    CoachableMomentOut,
    SegmentOut,
    SentimentOut,
    TranscribeResponse,
)
from app.services.coach_service import detect_coachable_moments
from app.services.sentiment_service import analyse_sentiment
from app.services.stt_service import transcribe_audio

router = APIRouter(prefix="/transcribe", tags=["STT"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "tmp/audio_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


@router.post(
    "",
    response_model=TranscribeResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe audio with diarization and coachable moment detection",
)
async def transcribe_endpoint(
    file:        UploadFile    = File(..., description="Audio file (WAV/MP3)"),
    call_id:     Optional[str] = Form(None),
    agent_id:    Optional[str] = Form(None),
    customer_id: Optional[str] = Form(None),
    db:          Session       = Depends(get_db),
):
    # ── Validate file extension ───────────────────────────────────────────────
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    call_id = call_id or str(uuid.uuid4())

    # ── Save uploaded file ────────────────────────────────────────────────────
    audio_path = UPLOAD_DIR / f"{call_id}{suffix}"
    try:
        with audio_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Saved audio file: {audio_path}")
    except Exception as exc:
        logger.error(f"Failed to save audio: {exc}")
        raise HTTPException(status_code=500, detail="Could not save audio file.")
    finally:
        await file.close()

    # ── STT Transcription ─────────────────────────────────────────────────────
    try:
        stt_result = transcribe_audio(str(audio_path))
    except Exception as exc:
        logger.error(f"STT failed for call {call_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    # ── Sentiment on each segment (never crashes the whole request) ───────────
    for seg in stt_result["segments"]:
        try:
            seg["sentiment"] = analyse_sentiment(seg["text"])
        except Exception as exc:
            logger.warning(f"Sentiment skipped for segment: {exc}")
            seg["sentiment"] = {"label": "NEUTRAL", "score": 0.0}

    # ── Coachable moment detection ────────────────────────────────────────────
    moments = detect_coachable_moments(stt_result["segments"])

    # ── Persist to DB ─────────────────────────────────────────────────────────
    try:
        call_record = CallRecord(
            call_id        = call_id,
            agent_id       = agent_id,
            customer_id    = customer_id,
            audio_path     = str(audio_path),
            raw_transcript = stt_result["raw_transcript"],
            language       = stt_result["language"],
            duration_s     = stt_result["duration_s"],
        )
        db.add(call_record)
        db.flush()

        segment_orm_list = []
        for seg in stt_result["segments"]:
            s_obj = TranscriptSegment(
                call_id_fk      = call_record.id,
                speaker         = seg["speaker"],
                start_s         = seg["start_s"],
                end_s           = seg["end_s"],
                text            = seg["text"],
                sentiment       = seg["sentiment"]["label"],
                sentiment_score = seg["sentiment"]["score"],
            )
            db.add(s_obj)
            segment_orm_list.append(s_obj)

        moment_orm_list = []
        for m in moments:
            m_obj = CoachableMoment(
                call_id_fk  = call_record.id,
                moment_type = m["moment_type"],
                text        = m["text"],
                start_s     = m.get("start_s"),
                end_s       = m.get("end_s"),
                speaker     = m.get("speaker"),
                confidence  = m.get("confidence"),
            )
            db.add(m_obj)
            moment_orm_list.append(m_obj)

        db.commit()
        db.refresh(call_record)

    except Exception as exc:
        db.rollback()
        logger.error(f"DB persistence failed: {exc}")
        raise HTTPException(status_code=500, detail="Failed to save results to database.")

    # ── Build response ────────────────────────────────────────────────────────
    segments_out = [
        SegmentOut(
            speaker   = seg["speaker"],
            start_s   = seg["start_s"],
            end_s     = seg["end_s"],
            text      = seg["text"],
            sentiment = SentimentOut(**seg["sentiment"]),
        )
        for seg in stt_result["segments"]
    ]

    moments_out = [
        CoachableMomentOut(
            id          = m_obj.id,
            moment_type = m_obj.moment_type,
            text        = m_obj.text,
            start_s     = m_obj.start_s,
            end_s       = m_obj.end_s,
            speaker     = m_obj.speaker,
            confidence  = m_obj.confidence,
        )
        for m_obj in moment_orm_list
    ]

    return TranscribeResponse(
        call_id           = call_id,
        agent_id          = agent_id,
        customer_id       = customer_id,
        language          = stt_result["language"],
        duration_s        = stt_result["duration_s"],
        raw_transcript    = stt_result["raw_transcript"],
        segments          = segments_out,
        coachable_moments = moments_out,
    )