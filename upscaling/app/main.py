"""Thin FastAPI upscaler — parametrized scale, no OpenAI pretence.

Upscaling isn't part of the OpenAI image API and vLLM-Omni has no SR models, so
this is a deliberately small dedicated service: POST an image, get a bigger one
back. Pairs with the Z-Image generation service (models/z-image).

  POST /upscale   multipart 'file', optional 'scale' (2|4) and 'outscale' (float)
  GET  /health    readiness + available native scales
"""
from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from . import upscaler

# Native scales to load at boot (the rest load lazily on first request).
PRELOAD = [int(s) for s in os.environ.get("UPSCALER_PRELOAD", "4").split(",") if s.strip()]
DEFAULT_SCALE = int(os.environ.get("UPSCALER_DEFAULT_SCALE", "4"))

app = FastAPI(title="ai_services upscaler", version="0.2.0")
_ready = False


@app.on_event("startup")
def _startup() -> None:
    global _ready
    upscaler.preload(PRELOAD)
    _ready = True


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _ready else "loading",
        "native_scales": sorted(upscaler.NATIVE_WEIGHTS),
        "default_scale": DEFAULT_SCALE,
        "loaded": sorted(upscaler._models),
    }


@app.post("/upscale")
async def do_upscale(
    file: UploadFile = File(...),
    scale: int = Form(DEFAULT_SCALE),
    outscale: float | None = Form(None),
) -> Response:
    """Upscale the uploaded image.

    `scale`    — native model to run: 2 or 4 (default 4).
    `outscale` — optional arbitrary factor (e.g. 1.5, 3, 8); overrides `scale`,
                 resampled to the exact factor on top of the nearest native model.
    """
    if not _ready:
        raise HTTPException(status_code=503, detail="model still loading")
    try:
        img = Image.open(io.BytesIO(await file.read()))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad image: {exc}") from exc

    try:
        out, meta = upscaler.upscale(img, scale=scale, outscale=outscale)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    headers = {
        "X-Native-Scale": str(meta["native_scale"]),
        "X-Effective-Scale": str(meta["effective_scale"]),
        "X-Output-Size": f"{meta['output_size'][0]}x{meta['output_size'][1]}",
    }
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)
