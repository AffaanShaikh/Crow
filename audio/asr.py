"""
Automatic Speech Recognition (ASR) service,
- using faster-whisper, a CTranslate2-optimised reimplementation of OpenAI Whisper
that runs 2-4x faster than the original at equal accuracy, with lower memory usage

Two usage modes:-
  1. Streaming:
    - chunks arrive via WebSocket,
    - VAD detects utterance end,
    - Whisper transcribes the accumulated buffer
  2. Batch: a complete audio file transcribed in one call

Audio contract (must match frontend MediaRecorder config):-
  - Format : 16-bit PCM, mono
  - Sample rate : 16000 Hz
  - Encoding : little-endian signed int16
"""

from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from audio.vad import VADProcessor, pcm_bytes_to_float32, SAMPLE_RATE
from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

# Whisper model tiers: trade off speed vs accuracy
# tiny.en -> base.en -> small.en -> medium.en -> large-v3
# for real-time: base.en (fast) or small.en (balanced)
DEFAULT_MODEL_SIZE = "base.en"
MAX_BUFFER_SECONDS = 30 # hard cap to prevent unbounded accumulation


class TranscriptEventType(str, Enum):
    PARTIAL = "partial" # interim result (fast, less accurate)
    FINAL = "final" # utterance complete (accurate)
    ERROR = "error"


@dataclass
class TranscriptEvent:
    type: TranscriptEventType
    text: str
    confidence: float = 0.0
    duration_ms: float = 0.0
    language: str = "en"


@dataclass
class StreamingASRSession:
    """
    per-connection streaming state,
    each WebSocket connection gets its own session with:
      - An audio accumulation buffer
      - A VAD processor instance
      - An asyncio Queue for transcript events (consumed by the WS/WebSocket handler)
        : it's how the transcription thread communicates back to the WS handler without them needing to know about each other, the *producer-consumer pattern*
    """
    session_id: str
    buffer: bytearray = field(default_factory=bytearray)
    vad: VADProcessor = field(default_factory=VADProcessor)
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    is_active: bool = True
    total_audio_seconds: float = 0.0

    def push_chunk(self, pcm_bytes: bytes) -> None:
        self.buffer.extend(pcm_bytes)
        self.total_audio_seconds += len(pcm_bytes) / 2 / SAMPLE_RATE

    def pop_buffer(self) -> bytes:
        data = bytes(self.buffer)
        self.buffer.clear()
        return data

    def buffer_seconds(self) -> float:
        return len(self.buffer) / 2 / SAMPLE_RATE


class ASRService:
    """
    manages the faster-whisper model and handles transcription requests,
    model is loaded once at startup and shared across all sessions.
    Transcription is CPU/GPU bound: it runs in a thread pool to avoid blocking the asyncio event loop
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE, # Whisper model variant (tiny.en / base.en / small.en / medium.en / large-v3) 
        device: str = "auto", # 'cpu', 'cuda', or 'auto' (auto-detects GPU) 
        compute_type: str = "auto", # 'int8' (fastest/smallest), 'float16' (GPU), 'float32' (CPU accurate), 'auto' picks sensibly based on device
    ) -> None: 
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._sessions: dict[str, StreamingASRSession] = {}
        self._lock = asyncio.Lock()
        log.info("asr_service_created", model=model_size, device=device)

    # lifecycle of asr
    async def load_model(self) -> None:
        """loads Whisper model (blocking: run once at startup)"""
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)
        elapsed = time.perf_counter() - t0
        log.info("asr_model_loaded", model=self.model_size, elapsed_s=round(elapsed, 2))

    def _load_model_sync(self) -> None:
        from faster_whisper import WhisperModel
        device = self.device
        compute_type = self.compute_type
        # auto selecting sensible defaults for computation
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        self._model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
        )
        log.info("faster_whisper_loaded", device=device, compute_type=compute_type)

    # streaming session management
    def create_session(self, session_id: str) -> StreamingASRSession:
        session = StreamingASRSession(session_id=session_id)
        self._sessions[session_id] = session
        log.info("asr_session_created", session_id=session_id)
        return session

    def get_session(self, session_id: str) -> StreamingASRSession | None:
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            session.is_active = False
            log.info("asr_session_closed", session_id=session_id,
                     total_audio_s=round(session.total_audio_seconds, 2))

    # streaming ingestion
    async def ingest_chunk(self, session_id: str, pcm_bytes: bytes) -> None:
        """
        ingests raw PCM chunk from the WebSocket stream,
        VAD logic:
          - If chunk contains speech                    -> accumulate in buffer
          - If silence follows speech (utterance ended) -> transcribe buffer
          - If buffer exceeds MAX_BUFFER_SECONDS        -> force transcribe (safety valve)
        """
        session = self._sessions.get(session_id)
        if not session or not session.is_active:
            return

        is_speech = session.vad.is_speech(pcm_bytes)

        if is_speech:
            session.push_chunk(pcm_bytes)
        elif session.vad.just_ended() and len(session.buffer) > 0: # i.e. utterance completed, transcribe what we have
            audio_data = session.pop_buffer()
            session.vad.reset()
            await self._transcribe_and_emit(session, audio_data, is_final=True)

        elif session.buffer_seconds() > MAX_BUFFER_SECONDS: # i.e. if buffer grows too large, force transcription (safety valve) 
            log.warning("asr_buffer_overflow", session_id=session_id)
            audio_data = session.pop_buffer()
            await self._transcribe_and_emit(session, audio_data, is_final=True)

    # transcription:-
    async def _transcribe_and_emit(
        self,
        session: StreamingASRSession,
        pcm_bytes: bytes,
        is_final: bool = True,
    ) -> None:
        """runs Whisper in thread pool, emits result to session queue"""
        if len(pcm_bytes) < 1600: # i.e. if < 50 ms: skip noise
            return
        try:
            event = await self._transcribe_audio(pcm_bytes, is_final=is_final)
            if event.text.strip():
                await session.event_queue.put(event)
                log.info("asr_transcript", session_id=session.session_id,
                         text_preview=event.text[:60], final=is_final)
        except Exception as exc:
            log.exception("asr_transcription_error", error=str(exc))
            await session.event_queue.put(
                TranscriptEvent(type=TranscriptEventType.ERROR, text=str(exc))
            )

    async def _transcribe_audio(self, pcm_bytes: bytes, is_final: bool) -> TranscriptEvent:
        """offloading blocking Whisper inference to thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, # means "use the default thread pool"
            self._transcribe_sync,
            pcm_bytes,
            is_final,
        )

    def _transcribe_sync(self, pcm_bytes: bytes, is_final: bool) -> TranscriptEvent:
        """
        Blocking Whisper transcription, called only from thread pool

        faster-whisper returns a generator of segments with word-level
        timestamps, confidence scores, and language detection
        """
        if self._model is None:
            return TranscriptEvent(
                type=TranscriptEventType.ERROR,
                text="ASR model not loaded",
            )

        t0 = time.perf_counter()
        audio_f32 = pcm_bytes_to_float32(pcm_bytes)

        segments, info = self._model.transcribe(
            audio_f32,
            beam_size=5 if is_final else 1,     # faster beam for partial
            vad_filter=False,                   # already did our own VAD
            language="en",                      # remove for multilingual
            condition_on_previous_text=False,   # prevents hallucination loops
            temperature=0.0,                    # greedy for speed
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
        )

        texts = []
        avg_confidence = 0.0
        seg_count = 0
        for segment in segments:
            texts.append(segment.text.strip())
            avg_confidence += segment.avg_logprob
            seg_count += 1

        text = " ".join(texts).strip()
        confidence = float(np.exp(avg_confidence / max(seg_count, 1)))
        duration_ms = (time.perf_counter() - t0) * 1000

        log.debug("asr_inference_done",
                  text_len=len(text), duration_ms=round(duration_ms, 1),
                  confidence=round(confidence, 3))

        return TranscriptEvent(
            type=TranscriptEventType.FINAL if is_final else TranscriptEventType.PARTIAL,
            text=text,
            confidence=confidence,
            duration_ms=duration_ms,
            language=info.language,
        )

    # batch api,
    async def transcribe_file(self, audio_path: Path) -> str:
        """
        Transcribes an audio file (any format supported by ffmpeg)
        Returns:
            the full transcript as a string
        """
        loop = asyncio.get_event_loop()
        event = await loop.run_in_executor(None, self._transcribe_file_sync, audio_path)
        return event.text

    def _transcribe_file_sync(self, audio_path: Path) -> TranscriptEvent:
        if self._model is None:
            return TranscriptEvent(type=TranscriptEventType.ERROR, text="Model not loaded")
        segments, info = self._model.transcribe(str(audio_path), beam_size=5)
        text = " ".join(s.text.strip() for s in segments)
        return TranscriptEvent(
            type=TranscriptEventType.FINAL,
            text=text,
            language=info.language,
        )

    @property
    def is_ready(self) -> bool:
        return self._model is not None


# Singleton management:-
_asr_service: ASRService | None = None

def get_asr_service() -> ASRService:
    if _asr_service is None:
        raise RuntimeError("ASRService not initialised.")
    return _asr_service

def init_asr_service() -> ASRService:
    global _asr_service
    cfg = get_settings()
    _asr_service = ASRService(
        model_size=cfg.asr_model_size,
        device=cfg.asr_device,
        compute_type=cfg.asr_compute_type,
    )
    return _asr_service
