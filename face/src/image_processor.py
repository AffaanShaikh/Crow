"""
image_processor.py — Avatar Image Processing Service
-----------------------------------------------------
Runs as a standalone FastAPI service on port 8001.
Receives an image (URL, local path, or base64), removes its background
via rembg, and returns the result as a base64-encoded PNG.

Usage:
    pip install fastapi uvicorn rembg pillow httpx
    uvicorn image_processor:app --host 0.0.0.0 --port 8001 --reload

Endpoint:
    POST /process-image
    Body: { "image_url": "...", "image_path": "...", "image_base64": "..." }
    Response: { "image": "<base64 PNG>", "format": "png", "source": "..." }
"""

import base64
import io
import logging
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from rembg import remove, new_session

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("image_processor")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Avatar Image Processor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-load the rembg session once so the first request is fast
log.info("Loading rembg model (u2net)…")
_SESSION = new_session("u2net")
log.info("rembg model ready.")


# ── request / response models ─────────────────────────────────────────────────
class ImageRequest(BaseModel):
    image_url: str | None = None
    image_path: str | None = None
    image_base64: str | None = None


class ImageResponse(BaseModel):
    image: str          # base64-encoded PNG
    format: str = "png"
    source: str         # which input was used
    elapsed_ms: float


# ── helpers ───────────────────────────────────────────────────────────────────
def _remove_bg(raw_bytes: bytes) -> bytes:
    """Run rembg background removal and return PNG bytes."""
    result = remove(raw_bytes, session=_SESSION)
    # Ensure output is a proper RGBA PNG
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "image_processor"}


@app.post("/process-image", response_model=ImageResponse)
async def process_image(req: ImageRequest):
    t0 = time.perf_counter()
    raw: bytes | None = None
    source: str = ""

    # ── 1. Resolve input ──────────────────────────────────────────────────────
    if req.image_base64:
        source = "base64"
        log.info("Processing image from base64 input…")
        try:
            raw = base64.b64decode(req.image_base64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}")

    elif req.image_url:
        source = "url"
        log.info("Fetching image from URL: %s", req.image_url)
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
                resp = await client.get(req.image_url)
                resp.raise_for_status()
                raw = resp.content
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Fetch failed: {exc}")

    elif req.image_path:
        source = "path"
        path = os.path.abspath(req.image_path)
        log.info("Reading image from path: %s", path)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        with open(path, "rb") as fh:
            raw = fh.read()

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide one of: image_url, image_path, or image_base64",
        )

    # ── 2. Remove background ──────────────────────────────────────────────────
    log.info("Removing background (source=%s, size=%d bytes)…", source, len(raw))
    try:
        processed = _remove_bg(raw)
    except Exception as exc:
        log.error("rembg failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Background removal failed: {exc}")

    elapsed = (time.perf_counter() - t0) * 1000
    log.info("Done — %.1f ms, output size=%d bytes", elapsed, len(processed))

    return ImageResponse(
        image=base64.b64encode(processed).decode(),
        format="png",
        source=source,
        elapsed_ms=round(elapsed, 1),
    )
