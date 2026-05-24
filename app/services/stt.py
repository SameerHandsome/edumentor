"""
STT service — faster-whisper (base, CPU, int8).
Transcribes audio bytes to text. Singleton model loaded on first call.
"""

from __future__ import annotations

import functools
import os
import tempfile

import structlog
from prometheus_client import Histogram

from app.core.config import settings

logger = structlog.get_logger(__name__)

STT_LATENCY = Histogram(
    "stt_latency_seconds", "Whisper transcription latency", buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)


@functools.lru_cache(maxsize=1)
def _load_whisper():
    from faster_whisper import WhisperModel

    model = WhisperModel(
        settings.WHISPER_MODEL,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
    )
    logger.info("whisper_loaded", model=settings.WHISPER_MODEL)
    return model


async def transcribe(audio_bytes: bytes, language: str = "en") -> str:
    """
    Transcribe raw audio bytes (WAV/MP3/OGG) to text.
    Returns empty string on failure.
    """
    import time

    start = time.perf_counter()
    try:
        model = _load_whisper()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        segments, _ = model.transcribe(tmp_path, language=language, beam_size=5)
        text = " ".join(s.text for s in segments).strip()
        os.unlink(tmp_path)
        duration = time.perf_counter() - start
        STT_LATENCY.observe(duration)
        logger.info("transcription_done", chars=len(text), duration_ms=round(duration * 1000))
        return text
    except Exception as exc:
        logger.error("transcription_failed", error=str(exc))
        return ""
