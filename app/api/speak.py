# app/api/speak.py

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from loguru import logger

from app.schemas.request_response import SpeakRequest
from app.services.tts_service import synthesize_speech

router = APIRouter(prefix="/speak", tags=["TTS"])


@router.post(
    "",
    summary="Convert text to speech and return an MP3 file",
    response_class=FileResponse,
)
async def speak_endpoint(payload: SpeakRequest):
    """
    Accepts JSON body: {"text": "...", "language": "en", "slow": false}
    Returns: audio/mpeg (MP3 file download)
    """
    try:
        audio_path = synthesize_speech(
            text     = payload.text,
            language = payload.language,
            slow     = payload.slow,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.error(f"TTS synthesis error: {exc}")
        raise HTTPException(status_code=500, detail="TTS synthesis failed.")

    return FileResponse(
        path         = audio_path,
        media_type   = "audio/mpeg",
        filename     = "speech.mp3",
    )