"""
Voice Activity Detection (VAD),
- using silero VAD coz lightweight & accurate speech detector
- & faster-whisper ships with silero integration, but we expose it here
    as a standalone utility so the ASR class and the WebSocket handler
    can both query it independently

responsible for:
  - detecting speech boundaries in raw PCM audio
  - exposing is_speech() for real-time chunk filtering
  - exposing split_on_silence() for batch segmentation

Audio contract throughout this module:
  - Format : 16-bit PCM, mono
  - Sample rate : 16000 Hz (Whisper's native rate, no resampling needed)
  - Chunk size : 512 samples (32 ms at 16kHz, Silero's recommended window)
"""

from __future__ import annotations

import struct
import numpy as np
from dataclasses import dataclass
from typing import Iterator
from utils.logger import get_logger

log = get_logger(__name__)

# silero VAD chunk size req.(s)
SAMPLE_RATE = 16000
SILERO_CHUNK_SAMPLES = 512 # 32 ms
SILERO_CHUNK_BYTES = SILERO_CHUNK_SAMPLES * 2 # 16-bit = 2 bytes per sample
SPEECH_PAD_SECONDS = 0.3 # silence padding added around detected speech segments (in seconds)


@dataclass
class SpeechSegment:
    """contiguous block of audio containing speech"""
    start_sample: int
    end_sample: int
    audio: np.ndarray # float32 normalised [-1, 1]

    @property
    def duration_ms(self) -> float:
        return len(self.audio) / SAMPLE_RATE * 1000


class VADProcessor:
    """
    stateful VAD processor for real-time audio streams

    Usage (streaming)::

        vad = VADProcessor()
        for pcm_chunk in mic_stream():
            if vad.is_speech(pcm_chunk):
                # accumulate for transcription
                ...
            elif vad.just_ended():
                # silence after speech, trigger transcription
                ...

    Usage (batch)::
    
        vad = VADProcessor()
        for segment in vad.split_on_silence(full_audio_bytes):
            transcribe(segment.audio)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 600,
    ) -> None:
        """
        Args:
            threshold: silero confidence above which audio is considered as speech
            min_speech_ms: discard segments shorter than this (filters clicks/pops)
            min_silence_ms: silence duration needed to close a speech segment
        """
        self.threshold = threshold
        self.min_speech_samples = int(min_speech_ms / 1000 * SAMPLE_RATE)
        self.min_silence_samples = int(min_silence_ms / 1000 * SAMPLE_RATE)

        self._model = None # avoiding startup cost w/ lazy loading
        self._triggered = False # to see if: is currently inside speech?
        self._prev_was_speech = False

        log.debug("vad_init", threshold=threshold)

    # lazy model loading
    def _get_model(self):
        """load silero VAD on first use (avoiding import-time torch overhead)"""
        if self._model is None:
            try:
                import torch
                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                model.eval()
                self._model = model
                log.info("silero_vad loaded")
            except Exception as exc:
                log.warning("silero_vad unavailable", error=str(exc),
                            hint="pip install torch silero-vad")
                self._model = None
        return self._model

    # the real-time api
    def is_speech(self, pcm_bytes: bytes) -> bool:
        """
        return True if: PCM chunk (512 samples, 16-bit, 16kHz) contains speech
        fall back to True (pass-through) if: silero unavailable
        """
        model = self._get_model()
        if model is None:
            return True # handled degradation, let everything through

        try:
            import torch
            audio_f32 = pcm_bytes_to_float32(pcm_bytes)
            tensor = torch.tensor(audio_f32).unsqueeze(0)
            confidence = model(tensor, SAMPLE_RATE).item()
            is_speech = confidence >= self.threshold
            prev = self._prev_was_speech
            self._prev_was_speech = is_speech
            if is_speech and not self._triggered:
                self._triggered = True
            return is_speech
        except Exception as exc:
            log.warning("vad_inference_error", error=str(exc))
            return True

    def just_ended(self) -> bool:
        """
        Returns:
            true on the first silence chunk after a speech segment
        """
        if self._triggered and not self._prev_was_speech:
            self._triggered = False
            return True
        return False

    def reset(self) -> None:
        """resets state between utterances"""
        self._triggered = False
        self._prev_was_speech = False


    # batch api
    def split_on_silence(self, pcm_bytes: bytes) -> list[SpeechSegment]:
        """
        split a full audio recording into speech segments, discarding silence
        Args:
            pcm_bytes: Raw 16-bit PCM mono at 16kHz
        Returns:
            a List of SpeechSegment (may be empty if no speech was detected)
        """
        audio_f32 = pcm_bytes_to_float32(pcm_bytes)
        total_samples = len(audio_f32)
        pad = int(SPEECH_PAD_SECONDS * SAMPLE_RATE)

        model = self._get_model()
        if model is None:
            # no VAD available: return entire audio as one segment
            return [SpeechSegment(0, total_samples, audio_f32)]

        import torch

        segments: list[SpeechSegment] = []
        speech_start: int | None = None
        silence_count = 0

        for i in range(0, total_samples - SILERO_CHUNK_SAMPLES, SILERO_CHUNK_SAMPLES):
            chunk = audio_f32[i : i + SILERO_CHUNK_SAMPLES]
            tensor = torch.tensor(chunk).unsqueeze(0)
            confidence = model(tensor, SAMPLE_RATE).item()

            if confidence >= self.threshold:
                silence_count = 0
                if speech_start is None:
                    speech_start = max(0, i - pad)
            else:
                silence_count += SILERO_CHUNK_SAMPLES
                if speech_start is not None and silence_count >= self.min_silence_samples:
                    end = min(total_samples, i + pad)
                    seg_audio = audio_f32[speech_start:end]
                    if len(seg_audio) >= self.min_speech_samples:
                        segments.append(SpeechSegment(speech_start, end, seg_audio))
                    speech_start = None
                    silence_count = 0

        # capturing trailing segment
        if speech_start is not None:
            seg_audio = audio_f32[speech_start:]
            if len(seg_audio) >= self.min_speech_samples:
                segments.append(SpeechSegment(speech_start, total_samples, seg_audio))

        log.debug("vad_split", segments=len(segments), total_ms=total_samples / SAMPLE_RATE * 1000)
        return segments


# util.(s)
def pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    convert raw 16-bit signed PCM bytes -> float32 array normalised to [-1, 1],
    the format expected by Silero and Whisper 
    """
    n_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes[: n_samples * 2])
    return np.array(samples, dtype=np.float32) / 32768.0

def float32_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """convert float32 [-1, 1] numpy array -> raw 16-bit signed PCM bytes"""
    clipped = np.clip(audio, -1.0, 1.0)
    int16 = (clipped * 32767).astype(np.int16)
    return int16.tobytes()

def chunk_audio(pcm_bytes: bytes, chunk_samples: int = SILERO_CHUNK_SAMPLES) -> Iterator[bytes]:
    """yields a fixed-size byte chunks from a pcm buffer"""
    chunk_bytes = chunk_samples * 2 # 16-bit
    for i in range(0, len(pcm_bytes), chunk_bytes):
        yield pcm_bytes[i : i + chunk_bytes]
