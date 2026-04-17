"""
System-level wake-word detection.
Uses openwakeword - fully local, no API key, no internet.
    pip install openwakeword
    python -m openwakeword.download_models --model hey_mycroft
    OR in cmd:-
        py
        import openwakeword
        from openwakeword.model import Model
        openwakeword.utils.download_models("hey_mycroft")

The isolated process architecture:-
    Wake-word detection runs in a separate daemon thread with its own
    PyAudio input stream, completely independent from the ASR WebSocket.
    When a wake word is detected, it pushes an event into an asyncio.Queue
    that is consumed by the FastAPI SSE endpoint and forwarded to the browser.

Why a separate thread and not async?
  PyAudio callbacks are synchronous C-level callbacks. They cannot be made
  async, and blocking in the event loop would stall all HTTP responses.
  The thread -> asyncio.Queue bridge (asyncio.loop.call_soon_threadsafe) is
  the canonical solution: the sync thread never touches asyncio directly.

STATE MACHINE:-
  IDLE     -> (wake word detected)  -> DETECTED
  DETECTED -> (frontend notified)   -> IDLE

After a detection, the detector enters a 3-second cooldown to prevent
false repeats from the same utterance echoing.
Supported models:-
    "hey_mycroft"    - "Hey Mycroft"  - curr.
    "hey_jarvis"     - "Hey Jarvis"
    "alexa"          - "Alexa"
    "hey_rhasspy"    - "Hey Rhasspy"
    Later to be replaced by custom models (.onnx) (place in data/models/wake_word/ and loaded
    by setting WAKE_WORD_MODEL_PATH in .env to the absolute path)
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

# Samples per chunk fed to openwakeword.
# 1280 samples @ 16kHz = 80ms - openwakeword's native chunk size.
CHUNK_SAMPLES   = 1280
SAMPLE_RATE     = 16000
DETECTION_THRESHOLD  = 0.5     # confidence required to fire (0–1)
COOLDOWN_SECONDS     = 3.0     # silence after detection before re-arming

# How long the model-warm-up blocking call is acceptable (seconds)
MODEL_LOAD_TIMEOUT = 30.0


class WakeWordState(str, Enum):
    STOPPED  = "stopped"
    STARTING = "starting"
    LISTENING = "listening"
    DETECTED  = "detected"


@dataclass
class WakeEvent:
    model_name: str
    confidence: float
    timestamp: float


class WakeWordDetector:
    """
    Background thread that listens to the microphone and fires events
    when the configured wake word is detected.

    Thread safety: all public methods are safe to call from any thread.
    Event delivery: via asyncio.Queue - consumed by the SSE endpoint.
    """

    def __init__(
        self,
        model_name: str = "hey_mycroft",
        threshold: float = DETECTION_THRESHOLD,
        cooldown: float = COOLDOWN_SECONDS,
    ) -> None:
        self.model_name = model_name
        self.threshold  = threshold
        self.cooldown   = cooldown

        self._state   = WakeWordState.STOPPED
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop:  asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[WakeEvent] | None = None
        

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Start the background detection thread.
        Must be called with the running FastAPI event loop so thread-safe
        queue puts work correctly.
        """
        if self._state != WakeWordState.STOPPED:
            return

        self._loop  = loop
        self._queue = asyncio.Queue()
        self._stop_event.clear()
        self._state = WakeWordState.STARTING

        self._thread = threading.Thread(
            target=self._run,
            name="wake-word-detector",
            daemon=True,    # dies when the main process dies
        )
        self._thread.start()
        log.info("wake_word_starting", model=self.model_name)

    def stop(self) -> None:
        """Signal the detection thread to stop and wait for it."""
        if self._state == WakeWordState.STOPPED:
            return
        log.info("wake_word_stopping")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._state = WakeWordState.STOPPED

    @property
    def is_listening(self) -> bool:
        return self._state == WakeWordState.LISTENING

    # async event consumption

    async def events(self) -> AsyncIterator[WakeEvent]:
        """
        Async generator yielding WakeEvent objects as they are detected.
        Intended for the SSE endpoint to consume.
        """
        if self._queue is None:
            return
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                continue

    # thread detection

    def _run(self) -> None:
        """
        Blocking detection loop - runs in a daemon thread.

        Flow:
          1. Load the openwakeword model (once)
          2. Open PyAudio microphone stream
          3. Read chunks -> predict -> on threshold exceeded -> push event
          4. Sleep cooldown_seconds after each detection
          5. Exit when _stop_event is set
        """
        try:
            oww = self._load_model()
        except Exception as exc:
            log.error("wake_word_model_load_failed", error=str(exc))
            self._state = WakeWordState.STOPPED
            return

        try:
            audio = self._open_mic()
        except Exception as exc:
            log.error("wake_word_mic_failed", error=str(exc))
            self._state = WakeWordState.STOPPED
            return

        self._state = WakeWordState.LISTENING
        log.info("wake_word_listening", model=self.model_name, threshold=self.threshold)

        last_detection = 0.0

        try:
            while not self._stop_event.is_set():
                chunk = audio.read(CHUNK_SAMPLES, exception_on_overflow=False)

                import numpy as np
                pcm = np.frombuffer(chunk, dtype=np.int16)

                prediction: dict[str, Any] = oww.predict(pcm)

                for mdl_name, score in prediction.items():
                    if score >= self.threshold:
                        now = time.time()
                        if now - last_detection < self.cooldown:
                            continue
                        last_detection = now
                        event = WakeEvent(
                            model_name=mdl_name,
                            confidence=float(score),
                            timestamp=now,
                        )
                        self._push_event(event)
                        log.info(
                            "wake_word_detected",
                            model=mdl_name,
                            confidence=round(score, 3),
                        )

        except Exception as exc:
            if not self._stop_event.is_set():
                log.error("wake_word_loop_error", error=str(exc))
        finally:
            try: audio.stop_stream(); audio.close()
            except: pass
            try:
                import pyaudio
                pa = pyaudio.PyAudio(); pa.terminate()
            except: pass
            self._state = WakeWordState.STOPPED
            log.info("wake_word_stopped")

    def _load_model(self):
        """Load the openwakeword model. Blocking - runs in thread."""
        try:
            from openwakeword.model import Model
        except ImportError:
            raise RuntimeError(
                "openwakeword not installed. Run:\n"
                "  pip install openwakeword\n"
                "  python -m openwakeword.download_models --model hey_mycroft"
            )
        from config import get_settings
        settings = get_settings()

        # check for: custom model path in settings
        custom_path = getattr(settings, "wake_word_model_path", "")
        if custom_path:
            model = Model(wakeword_models=[custom_path], inference_framework="onnx")
        else:
            model = Model(wakeword_models=[self.model_name], inference_framework="onnx")

        log.info("wake_word_model_loaded", model=self.model_name)
        return model

    def _open_mic(self):
        """Open a PyAudio microphone stream. Blocking - runs in thread."""
        try:
            import pyaudio
        except ImportError:
            raise RuntimeError(
                "PyAudio not installed. Run: pip install pyaudio\n"
                "  Windows: pip install pyaudio\n"
                "  Linux: sudo apt-get install portaudio19-dev && pip install pyaudio\n"
                "  macOS: brew install portaudio && pip install pyaudio"
            )
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=CHUNK_SAMPLES,
        )
        stream.start_stream()
        return stream

    def _push_event(self, event: WakeEvent) -> None:
        """Thread-safe push of a WakeEvent into the asyncio queue."""
        if self._loop and self._queue and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)



_detector: WakeWordDetector | None = None


def get_wake_detector() -> WakeWordDetector:
    if _detector is None:
        raise RuntimeError("WakeWordDetector not initialised.")
    return _detector


def init_wake_detector() -> WakeWordDetector:
    global _detector
    from config import get_settings
    settings = get_settings()
    _detector = WakeWordDetector(
        model_name=getattr(settings, "wake_word_model", "hey_mycroft"),
        threshold=getattr(settings, "wake_word_threshold", 0.5),
    )
    return _detector