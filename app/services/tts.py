"""
TTS service — Coqui XTTS-v2 (local, streaming).
Synthesizes text to audio bytes (WAV). Singleton model loaded lazily.
"""

from __future__ import annotations

import functools
import io
import os
from collections.abc import AsyncIterator

import structlog
from prometheus_client import Histogram

from app.core.config import settings

logger = structlog.get_logger(__name__)

TTS_LATENCY = Histogram(
    "tts_latency_seconds", "Coqui TTS synthesis latency", buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 15.0]
)


@functools.lru_cache(maxsize=1)
def _load_tts():
    from TTS.api import TTS

    tts = TTS(model_name=settings.COQUI_MODEL_NAME, progress_bar=False)
    is_multi = getattr(tts, "is_multi_lingual", False)
    logger.info("coqui_tts_loaded", model=settings.COQUI_MODEL_NAME, multilingual=is_multi)
    return tts


def _is_multilingual() -> bool:
    """Return True only if the loaded model actually supports a language arg."""
    try:
        tts = _load_tts()
        return bool(getattr(tts, "is_multi_lingual", False))
    except Exception:
        return False


async def synthesize(text: str, language: str = "en") -> bytes:
    """
    Synthesize text to WAV bytes.
    Returns empty bytes on failure.

    The `language` kwarg is only forwarded when the loaded Coqui model is
    genuinely multi-lingual (e.g. xtts_v2). Mono-lingual models raise
    "Model is not multi-lingual but `language` is provided" — so we detect
    this at call time and drop the arg for those models.
    """
    import asyncio
    import time

    start = time.perf_counter()
    try:
        tts = _load_tts()
        speaker_wav = (
            settings.COQUI_SPEAKER_WAV if os.path.exists(settings.COQUI_SPEAKER_WAV) else None
        )
        # Use configured language default; fall back to the caller's value.
        lang = getattr(settings, "COQUI_LANGUAGE", language) or language
        multilingual = _is_multilingual()

        loop = asyncio.get_event_loop()
        buf = io.BytesIO()

        def _synth():
            kwargs: dict = {"text": text, "file_path": buf}
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
            if multilingual:
                kwargs["language"] = lang
            tts.tts_to_file(**kwargs)

        await loop.run_in_executor(None, _synth)
        audio_bytes = buf.getvalue()
        duration = time.perf_counter() - start
        TTS_LATENCY.observe(duration)
        logger.info(
            "tts_done",
            chars=len(text),
            audio_bytes=len(audio_bytes),
            duration_ms=round(duration * 1000),
            multilingual=multilingual,
        )
        return audio_bytes
    except Exception as exc:
        logger.error("tts_failed", error=str(exc))
        return b""


async def synthesize_stream(text: str, language: str = "en") -> AsyncIterator[bytes]:
    """
    Yield audio chunks for SSE streaming.
    Splits text into sentences and synthesizes each for lower TTFB.
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        if sentence.strip():
            chunk = await synthesize(sentence.strip(), language=language)
            if chunk:
                yield chunk
