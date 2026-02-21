# app/services/tts_service.py
"""
Text-to-Speech service using gTTS.
Returns path to a generated MP3 file.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from gtts import gTTS
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

OUTPUT_DIR = Path(os.getenv("TTS_OUTPUT_DIR", "/tmp/tts_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def synthesize_speech(
    text:     str,
    language: str = "en",
    slow:     bool = False,
) -> str:
    """
    Convert text to speech and save to a temp MP3 file.

    Args:
        text:     Input text to synthesize.
        language: BCP-47 language code (default 'en').
        slow:     Speak slowly (default False).

    Returns:
        Absolute path to the generated MP3 file.
    """
    if not text.strip():
        raise ValueError("Text for TTS cannot be empty.")

    file_name = f"{uuid.uuid4().hex}.mp3"
    out_path  = OUTPUT_DIR / file_name

    logger.info(f"Synthesizing TTS → {out_path}")
    tts = gTTS(text=text, lang=language, slow=slow)
    tts.save(str(out_path))
    logger.info("TTS synthesis complete.")
    return str(out_path)