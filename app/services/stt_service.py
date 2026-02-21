# app/services/stt_service.py
"""
Speech-to-Text service using OpenAI Whisper (local) with speaker
diarization via pyannote.audio.  Falls back gracefully if pyannote
is unavailable (e.g., no HuggingFace token provided).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import torch
import whisper
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")   # tiny|base|small|medium|large
HF_TOKEN           = os.getenv("HF_TOKEN", "")            # required for pyannote
_whisper_model     = None
_diarize_pipeline  = None


def _load_whisper() -> whisper.Whisper:
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _whisper_model


def _load_diarizer():
    """Load pyannote diarization pipeline (requires HF_TOKEN)."""
    global _diarize_pipeline
    if _diarize_pipeline is not None:
        return _diarize_pipeline
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set — speaker diarization disabled.")
        return None
    try:
        from pyannote.audio import Pipeline
        _diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _diarize_pipeline = _diarize_pipeline.to(torch.device(device))
        logger.info("Pyannote diarization pipeline loaded.")
    except Exception as exc:
        logger.warning(f"Could not load diarization pipeline: {exc}")
        _diarize_pipeline = None
    return _diarize_pipeline


# ── Diarization helpers ───────────────────────────────────────────────────────

def _diarize(audio_path: str) -> List[Tuple[float, float, str]]:
    """
    Returns list of (start_s, end_s, speaker_label).
    Falls back to single-speaker if pipeline unavailable.
    """
    pipeline = _load_diarizer()
    if pipeline is None:
        return []   # caller will assign single speaker

    try:
        diarization = pipeline(audio_path)
        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append((turn.start, turn.end, speaker))
        return result
    except Exception as exc:
        logger.error(f"Diarization failed: {exc}")
        return []


def _assign_speaker(
    seg_start: float,
    seg_end:   float,
    diar_spans: List[Tuple[float, float, str]],
) -> str:
    """Best-overlap speaker for a Whisper segment."""
    if not diar_spans:
        return "SPEAKER_00"

    best_speaker, best_overlap = "SPEAKER_00", 0.0
    for (d_start, d_end, speaker) in diar_spans:
        overlap = max(0.0, min(seg_end, d_end) - max(seg_start, d_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


# ── Main public function ──────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio and return structured result.

    Returns:
        {
          "language": str,
          "duration_s": float,
          "raw_transcript": str,
          "segments": [
              {"speaker": str, "start_s": float, "end_s": float, "text": str},
              ...
          ]
        }
    """
    logger.info(f"Starting transcription: {audio_path}")
    model = _load_whisper()

    # Whisper transcription with word-level timestamps
    result = model.transcribe(
        audio_path,
        verbose=False,
        word_timestamps=False,
        fp16=torch.cuda.is_available(),
    )

    language    = result.get("language", "en")
    raw_text    = result.get("text", "").strip()
    w_segments  = result.get("segments", [])

    # Estimate duration from last segment or audio file
    duration_s = w_segments[-1]["end"] if w_segments else 0.0

    # Speaker diarization
    diar_spans = _diarize(audio_path)

    segments = []
    for seg in w_segments:
        speaker = _assign_speaker(seg["start"], seg["end"], diar_spans)
        segments.append({
            "speaker": speaker,
            "start_s": round(seg["start"], 3),
            "end_s":   round(seg["end"],   3),
            "text":    seg["text"].strip(),
        })

    logger.info(f"Transcription done: {len(segments)} segments, lang={language}")
    return {
        "language":       language,
        "duration_s":     round(duration_s, 3),
        "raw_transcript": raw_text,
        "segments":       segments,
    }