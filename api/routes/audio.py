"""
Audio API routes.

WebSocket  /audio/asr/stream    - real-time speech-to-text
POST       /audio/tts           - text-to-speech (returns WAV)
GET        /audio/tts/stream    - text-to-speech SSE (sentence-by-sentence)
GET        /audio/voices        - list available TTS voices

WebSocket protocol for ASR:
  Client -> Server:  raw 16-bit PCM chunks (binary frames), 512-sample aligned, 16kHz
  Server -> Client:  JSON text frames matching TranscriptEventSchema

  Flow:
    1. Client opens WebSocket, passes ?session_id=<uuid>
    2. Client streams raw PCM chunks as binary frames
    3. Server runs VAD -> accumulates speech -> transcribes on silence
    4. Server sends JSON transcript events back
    5. Client sends final transcript as a chat message
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from audio.asr import ASRService, TranscriptEventType, get_asr_service
from audio.tts import TTSService, AudioEventType, get_tts_service
from config import get_settings, Settings
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/audio", tags=["audio"])


# ASR - WebSocket streaming
@router.websocket("/asr/stream")
async def asr_stream(
    websocket: WebSocket,
    session_id: str = Query(default_factory=lambda: str(uuid.uuid4())),
    asr: ASRService = Depends(get_asr_service),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    WebSocket endpoint for real-time speech-to-text.
    Binary frames in -> JSON transcript events out.

    Client sends: raw 16-bit PCM bytes at 16kHz
    Server sends: { "type": "final"|"partial"|"error", "text": "...", "confidence": 0.95 }

    The client should:
      1. Use MediaRecorder or AudioWorklet to capture mic at 16kHz mono
      2. Send 512-sample (1024-byte) chunks as binary WebSocket frames
      3. On receiving a "final" event, use the text as the chat input
    """
    if not settings.asr_enabled:
        await websocket.close(code=1008, reason="ASR is disabled")
        return

    if not asr.is_ready:
        await websocket.close(code=1011, reason="ASR model not loaded")
        return

    await websocket.accept()
    session = asr.create_session(session_id)
    log.info("asr_ws_connected", session_id=session_id)

    try:
        # Runs two coroutines concurrently:-
        #   reader - receives PCM chunks from client
        #   sender - forwards transcript events to client
        import asyncio
        await asyncio.gather(
            _asr_reader(websocket, asr, session_id),
            _asr_sender(websocket, session),
        )
    except WebSocketDisconnect:
        log.info("asr_ws_disconnected", session_id=session_id)
    except Exception as exc:
        log.exception("asr_ws_error", session_id=session_id, error=str(exc))
    finally:
        asr.close_session(session_id)

async def _asr_reader(websocket: WebSocket, asr: ASRService, session_id: str) -> None:
    """Receive raw PCM frames from the client and feed them to ASR."""
    while True:
        try:
            data = await websocket.receive_bytes()
            await asr.ingest_chunk(session_id, data)
        except WebSocketDisconnect:
            break
        except Exception as exc:
            log.warning("asr_reader_error", error=str(exc))
            break

async def _asr_sender(websocket: WebSocket, session) -> None:
    """
    Forwards transcript events from the ASR queue to the WebSocket client.
    Uses a polling loop (checks queue every 10ms) so it can be cancelled cleanly.
    """
    import asyncio
    while session.is_active:
        try:
            event = session.event_queue.get_nowait() # get_nowait() retrieves an item if one is available else raise empty queue instead of waiting
            payload = {
                "type": event.type.value,
                "text": event.text,
                "confidence": round(event.confidence, 3),
                "duration_ms": round(event.duration_ms, 1),
                "language": event.language,
            }
            await websocket.send_text(json.dumps(payload))
        except Exception:
            await asyncio.sleep(0.01) # queue empty - yield


# TTS - HTTP endpoints 
class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float | None = None


@router.post("/tts", summary="Synthesise speech, return WAV file")
async def synthesise_speech(
    req: TTSRequest,
    tts: TTSService = Depends(get_tts_service),
    settings: Settings = Depends(get_settings),
) -> Response:
    """
    synthesise the given text and return a WAV audio file
    Returns:
        a single WAV file (all sentences concatenated),
        for long text use /tts/stream instead
    """
    if not settings.tts_enabled:
        raise HTTPException(status_code=503, detail="TTS is disabled")
    if not tts.is_ready:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    pcm_bytes = await tts.synthesise(req.text, req.voice, req.speed)

    # wrap PCM in WAV container
    from audio.tts import AudioChunk, AudioEventType
    chunk = AudioChunk(
        type=AudioEventType.CHUNK,
        audio_bytes=pcm_bytes,
        sentence_index=0,
    )
    wav_bytes = chunk.to_wav_bytes()

    log.info("tts_http_response", text_len=len(req.text), wav_bytes=len(wav_bytes))
    return Response(content=wav_bytes, media_type="audio/wav")


@router.post("/tts/stream", summary="Synthesise speech, stream sentence-by-sentence")
async def synthesise_speech_stream(
    req: TTSRequest,
    tts: TTSService = Depends(get_tts_service),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    Stream synthesised audio chunks via multipart/mixed response.

    Each part is a WAV file for one sentence. The client can start
    playing the first sentence while later ones are still being synthesised,
    reducing perceived latency significantly for long responses.
    """
    if not settings.tts_enabled:
        raise HTTPException(status_code=503, detail="TTS is disabled")

    # Each chunk starts with --BOUNDARY and the final chunk ends with --BOUNDARY--
    BOUNDARY = "audio-boundary"

    async def generate():
        async for chunk in tts.synthesise_streaming(req.text, req.voice, req.speed):
            if chunk.type == AudioEventType.CHUNK and chunk.audio_bytes:
                wav = chunk.to_wav_bytes()
                header = (
                    f"--{BOUNDARY}\r\n"
                    f"Content-Type: audio/wav\r\n"
                    f"Content-Length: {len(wav)}\r\n"
                    f"X-Sentence-Index: {chunk.sentence_index}\r\n"
                    f"\r\n"
                ).encode()
                yield header + wav + b"\r\n"
        yield f"--{BOUNDARY}--\r\n".encode()

    return StreamingResponse(
        generate(),
        media_type=f"multipart/mixed; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/voices", summary="List available TTS voices")
async def list_voices(settings: Settings = Depends(get_settings)) -> dict:
    """Return available Kokoro voice IDs and their descriptions."""
    from audio.tts import TTSService
    return {
        "voices": TTSService.VOICES,
        "default": settings.tts_voice,
        "enabled": settings.tts_enabled,
    }
