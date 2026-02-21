# app/services/sentiment_service.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from loguru import logger

# ── Point to local downloaded cache ──────────────────────────────────────────
CACHE_DIR = Path("./hf_cache").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TRANSFORMERS_CACHE"]    = str(CACHE_DIR)
os.environ["HF_HOME"]               = str(CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR)

# ── Fix broken SSL_CERT_FILE ──────────────────────────────────────────────────
_ssl_cert = os.environ.get("SSL_CERT_FILE", "")
if _ssl_cert and not Path(_ssl_cert).exists():
    del os.environ["SSL_CERT_FILE"]

_ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE", "")
if _ca_bundle and not Path(_ca_bundle).exists():
    del os.environ["REQUESTS_CA_BUNDLE"]

MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)

_pipeline       = None
_load_attempted = False


def _load_pipeline():
    global _pipeline, _load_attempted

    if _pipeline is not None:
        return _pipeline

    if _load_attempted:
        return None

    _load_attempted = True

    try:
        from transformers import pipeline as hf_pipeline

        logger.info(f"Loading sentiment model from local cache: {CACHE_DIR}")

        _pipeline = hf_pipeline(
            task             = "sentiment-analysis",
            model            = MODEL_NAME,
            truncation       = True,
            max_length       = 512,
            local_files_only = True,   # ← uses downloaded cache, no internet/SSL needed
            cache_dir        = str(CACHE_DIR),
        )

        logger.info("✅ Sentiment model loaded successfully.")

    except Exception as exc:
        logger.warning(f"⚠️  Sentiment model failed: {exc} — retrying with internet...")

        # fallback: try with internet
        try:
            from transformers import pipeline as hf_pipeline
            _pipeline = hf_pipeline(
                task      = "sentiment-analysis",
                model     = MODEL_NAME,
                truncation = True,
                max_length = 512,
                cache_dir  = str(CACHE_DIR),
            )
            logger.info("✅ Sentiment model loaded via internet.")
        except Exception as exc2:
            logger.warning(f"⚠️  Sentiment fully failed: {exc2} — returning NEUTRAL for all.")
            _pipeline = None

    return _pipeline


def analyse_sentiment(text: str) -> Dict[str, float | str]:
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    pipe = _load_pipeline()
    if pipe is None:
        return {"label": "NEUTRAL", "score": 0.0}

    try:
        result = pipe(text[:512])[0]
        label  = result["label"].upper()
        score  = round(result["score"], 4)
        if score < 0.65:
            label = "NEUTRAL"
        return {"label": label, "score": score}
    except Exception as exc:
        logger.warning(f"Sentiment inference error: {exc}")
        return {"label": "NEUTRAL", "score": 0.0}


def analyse_batch(texts: List[str]) -> List[Dict[str, float | str]]:
    return [analyse_sentiment(t) for t in texts]