"""
Wake-word SSE endpoint addition for audio routes:-
    GET /audio/wake/events  - SSE stream of wake-word detection events
    GET /audio/wake/status  - current detector state

The frontend connects to /audio/wake/events using EventSource and subscribes
to detection events. When the browser receives "wake_detected", it starts ASR.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from audio.wake_word import WakeWordDetector, WakeEvent, get_wake_detector
from config import get_settings, Settings
from utils.logger import get_logger

log = get_logger(__name__)


def register_wake_routes(router: APIRouter) -> None:
    """
    Register wake-word routes onto an existing APIRouter.
    Called from api/routes/audio.py:

        from api.routes.wake_routes import register_wake_routes
        register_wake_routes(router)
    """

    @router.get("/wake/events", include_in_schema=True)
    async def wake_events(settings: Settings = Depends(get_settings)):
        """
        SSE stream of wake-word detection events.
        The frontend subscribes here with EventSource.

        Events:
          {"type": "wake_listening"}              - detector is running
          {"type": "wake_detected", "model": "..."} - wake word heard
          {"type": "wake_stopped"}                - detector stopped

        The connection is long-lived. The client should reconnect on error.
        """
        if not getattr(settings, "wake_word_enabled", False):
            async def _disabled():
                yield f"data: {json.dumps({'type': 'wake_stopped', 'reason': 'disabled'})}\n\n"
            return StreamingResponse(_disabled(), media_type="text/event-stream")

        detector = get_wake_detector()

        async def _event_stream():
            # Send initial state
            state = "wake_listening" if detector.is_listening else "wake_stopped"
            yield f"data: {json.dumps({'type': state})}\n\n"

            async for event in detector.events():
                payload = {
                    "type": "wake_detected",
                    "model": event.model_name,
                    "confidence": round(event.confidence, 3),
                }
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/wake/status")
    async def wake_status(settings: Settings = Depends(get_settings)) -> dict:
        """Return current wake-word detector state."""
        if not getattr(settings, "wake_word_enabled", False):
            return {"enabled": False, "state": "disabled"}
        try:
            detector = get_wake_detector()
            return {
                "enabled":  True,
                "state":    "listening" if detector.is_listening else "stopped",
                "model":    detector.model_name,
                "threshold": detector.threshold,
            }
        except RuntimeError:
            return {"enabled": True, "state": "not_initialised"}