# app/api/replay.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from loguru import logger
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import CoachableMoment
from app.schemas.request_response import ReplayRequest
from app.services.tts_service import synthesize_speech

router = APIRouter(prefix="/replay", tags=["Coachable Moments"])


@router.post(
    "",
    summary="Replay a coachable moment via TTS",
    response_class=FileResponse,
)
async def replay_endpoint(
    payload: ReplayRequest,
    db:      Session = Depends(get_db),
):
    """
    Looks up a coachable moment by ID, synthesizes it via TTS,
    and returns the audio file for playback / agent training.
    """
    moment: CoachableMoment | None = db.get(CoachableMoment, payload.moment_id)

    if not moment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Coachable moment '{payload.moment_id}' not found.",
        )

    # Construct a spoken coaching prompt
    coaching_text = (
        f"Coachable moment type: {moment.moment_type.replace('_', ' ')}. "
        f"The {moment.speaker or 'speaker'} said: {moment.text}"
    )

    try:
        audio_path = synthesize_speech(text=coaching_text)
    except Exception as exc:
        logger.error(f"TTS replay error: {exc}")
        raise HTTPException(status_code=500, detail="TTS replay failed.")

    logger.info(f"Replaying moment {payload.moment_id}: {moment.moment_type}")

    return FileResponse(
        path       = audio_path,
        media_type = "audio/mpeg",
        filename   = f"replay_{payload.moment_id}.mp3",
    )


@router.get(
    "/list/{call_id}",
    summary="List all coachable moments for a call",
    tags=["Coachable Moments"],
)
async def list_moments(call_id: str, db: Session = Depends(get_db)):
    """List all coachable moments stored for a given call_id."""
    from app.db.models import CallRecord
    record = db.query(CallRecord).filter(CallRecord.call_id == call_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Call not found.")
    moments = record.coachable_moments
    return [
        {
            "id":          m.id,
            "moment_type": m.moment_type,
            "text":        m.text,
            "start_s":     m.start_s,
            "end_s":       m.end_s,
            "speaker":     m.speaker,
            "confidence":  m.confidence,
        }
        for m in moments
    ]