"""
Text-to-Speech (TTS) service,
- using Kokoro-82M, lightweight (82M parameter) neural TTS model that runs
entirely locally on CPU, producing near-VITS-quality speech.

Streaming strategy:-
  rather than waiting for the full response to synthesise, we:-
    1. split the text into sentences as soon as they arrive
    2. synthesise each sentence independently (200-400 ms each)
    3. stream each audio chunk to the client via WebSocket
  This gives time-to-first-audio of ~300 ms instead of waiting for the entire paragraph to be synthesised

Audio output contract:
  - Format      : 16-bit PCM, mono
  - Sample rate : 24000 Hz (Kokoro's native output, do NOT resample)
  - Encoding    : base64-encoded for JSON transport, or raw bytes for WebSocket
"""

from __future__ import annotations

import asyncio
import io
import re
import time
import wave
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

import numpy as np

from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

KOKORO_SAMPLE_RATE = 24_000 # Kokoro outputs at 24kHz
DEFAULT_VOICE = "af_heart"  # female voice
DEFAULT_SPEED = 1.0


class AudioEventType(str, Enum):
    CHUNK = "audio_chunk"   # PCM bytes for a sentence
    DONE = "audio_done"     # all chunks sent
    ERROR = "audio_error"


@dataclass
class AudioChunk:
    """a synthesised audio segment ready for playback"""
    type: AudioEventType
    audio_bytes: bytes # raw 16-bit PCM at 24kHz
    sentence_index: int
    sample_rate: int = KOKORO_SAMPLE_RATE
    error: str | None = None

    def to_wav_bytes(self) -> bytes:
        """wraps raw PCM in a WAV container for universal playback"""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.audio_bytes)
        return buf.getvalue()


class TTSService:
    """
    Manages the Kokoro pipeline and exposes streaming synthesis.

    The Kokoro pipeline is stateless between calls, so we hold one instance and call it concurrently.
    Multiple simultaneous synthesis requests run in the thread pool without blocking each other.
    """
    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED,
        lang_code: str = "a", # 'a' = American English
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self._pipeline = None
        log.info("tts_service_created", voice=voice, speed=speed)

    # tts lifecycle
    async def load_model(self) -> None:
        """loads Kokoro pipeline (runs once at startup)"""
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)
        elapsed = time.perf_counter() - t0
        log.info("tts_model_loaded", elapsed_s=round(elapsed, 2))

    def _load_sync(self) -> None:
        try:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code=self.lang_code)
            log.info("kokoro_pipeline_ready", lang=self.lang_code)
        except ImportError as exc:
            log.warning("kokoro_unavailable", error=str(exc),
                        hint="pip install kokoro soundfile")

    # streaming api (public)
    async def synthesise_streaming(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """
        Async generator that yields one AudioChunk per sentence, each chunk is ready to play immediately,
        so the client can start audio playback before the full text is synthesised 

        Usage::

            async for chunk in tts.synthesise_streaming(long_text):
                await websocket.send_bytes(chunk.audio_bytes)
        """
        sentences = split_into_sentences(text)
        log.debug("tts_streaming_start", sentences=len(sentences), chars=len(text))

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            try:
                chunk = await self._synthesise_sentence(
                    sentence, voice or self.voice, speed or self.speed, i
                )
                yield chunk
            except Exception as exc:
                log.exception("tts_chunk_error", sentence_idx=i, error=str(exc))
                yield AudioChunk(
                    type=AudioEventType.ERROR,
                    audio_bytes=b"",
                    sentence_index=i,
                    error=str(exc),
                )

    async def synthesise(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> bytes:
        """
        Synthesise entire text, return concatenated PCM bytes.
        Use for short strings (button labels, confirmations).
        For long text use synthesise_streaming().
        """
        chunks: list[bytes] = []
        async for chunk in self.synthesise_streaming(text, voice, speed):
            if chunk.type == AudioEventType.CHUNK:
                chunks.append(chunk.audio_bytes)
        return b"".join(chunks) # constructed waveform

    # internal util.(s)
    async def _synthesise_sentence(
        self,
        text: str,
        voice: str,
        speed: float,
        index: int,
    ) -> AudioChunk:
        """Runs Kokoro synthesis in thread pool for one sentence."""
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None, self._synthesise_sync, text, voice, speed
        )
        return AudioChunk(
            type=AudioEventType.CHUNK,
            audio_bytes=audio_bytes,
            sentence_index=index,
        )

    def _synthesise_sync(self, text: str, voice: str, speed: float) -> bytes:
        """
        Blocking Kokoro synthesis. Called from thread pool only.

        Kokoro generates audio as float32 numpy arrays. We collect
        all generator output and convert to 16-bit PCM bytes.
        """
        if self._pipeline is None:
            raise RuntimeError("TTS pipeline not loaded.")

        t0 = time.perf_counter()
        audio_segments: list[np.ndarray] = []

        # Kokoro pipeline is a generator, each iteration yields (graphemes, phonemes, audio)
        for _, _, audio in self._pipeline(text, voice=voice, speed=speed):
            if audio is not None and len(audio) > 0:
                audio_segments.append(audio)

        if not audio_segments:
            return b""

        full_audio = np.concatenate(audio_segments)
        pcm_bytes = float32_to_pcm16(full_audio)
        duration_ms = len(full_audio) / KOKORO_SAMPLE_RATE * 1000 
        elapsed_ms = (time.perf_counter() - t0) * 1000

        log.debug("tts_sentence_done",
                  text_preview=text[:40],
                  audio_ms=round(duration_ms),
                  synth_ms=round(elapsed_ms),
                  # Real-Time Factor (RTF) logged for each sentence, RTF < 1.0 means faster than real-time, which is our goal
                  rtf=round(elapsed_ms / max(duration_ms, 1), 3)) # RTF = synthesis_time / audio_duration, Kokoro achieves ~0.1-0.3 RTF on CPU

        return pcm_bytes

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    # avail. voices
    VOICES: dict[str, str] = {
        "af_heart":     "Female (warm, natural)",   # default
        "af_bella":     "Female (expressive)",
        "af_nicole":    "Female (gentle)",
        "am_adam":      "Male (clear)",
        "am_michael":   "Male (deep)",
        "bf_emma":      "Female British",
        "bm_george":    "Male British",
    }


# sentence splitting
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
_MIN_SENTENCE_CHARS = 15 # coz shouldn't synthesise tiny fragments separately


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into synthesis units at sentence boundaries.
    Short fragments are merged with the next sentence to avoid synthesising single words (which would sound unnatural and waste overhead).
    
    Example:
    
        "Hello! How are you? I'm fine." -> ["Hello!", "How are you?", "I'm fine."]
    """
    raw = _SENTENCE_END.split(text.strip())
    merged: list[str] = []
    buffer = ""

    for sentence in raw:
        sentence = sentence.strip()
        if not sentence:
            continue
        buffer = (buffer + " " + sentence).strip() if buffer else sentence
        if len(buffer) >= _MIN_SENTENCE_CHARS:
            merged.append(buffer)
            buffer = ""

    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer # merge short tail
        else:
            merged.append(buffer)

    return merged


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] numpy array -> raw 16-bit signed PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


# singleton manaaagemaunt
_tts_service: TTSService | None = None
 
def get_tts_service() -> TTSService:
    if _tts_service is None:
        raise RuntimeError("TTSService not initialised.")
    return _tts_service
 
def init_tts_service() -> TTSService:
    global _tts_service
    cfg = get_settings()
    _tts_service = TTSService(
        voice=cfg.tts_voice,
        speed=cfg.tts_speed,
        lang_code=cfg.tts_lang_code,
    )
    return _tts_service
