"""Thin FastAPI upscaler — parametrized scale, no OpenAI pretence.

Upscaling isn't part of the OpenAI image API and vLLM-Omni has no SR models, so
this is a deliberately small dedicated service: POST an image, get a bigger one
back. Pairs with the Z-Image generation service (models/z-image).

  POST /upscale       multipart 'file', optional 'scale' (2|4) and 'outscale' (float)
  POST /upscale/json  JSON b64 in / b64 out, to chain with the generation
                      services without a multipart round-trip through disk
  GET  /health        readiness + available native scales
"""
from __future__ import annotations

import base64
import binascii
import io
import os
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from . import upscaler

# Native scales to load at boot (the rest load lazily on first request).
PRELOAD = [int(s) for s in os.environ.get("UPSCALER_PRELOAD", "4").split(",") if s.strip()]
DEFAULT_SCALE = int(os.environ.get("UPSCALER_DEFAULT_SCALE", "4"))

app = FastAPI(title="ai_services upscaler", version="0.3.0")
_ready = False


@app.on_event("startup")
def _startup() -> None:
    global _ready
    upscaler.preload(PRELOAD)
    _ready = True


def _decode_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad image: {exc}") from exc
    return img


def _run(img: Image.Image, scale: int, outscale: float | None) -> tuple[Image.Image, dict]:
    if not _ready:
        raise HTTPException(status_code=503, detail="model still loading")
    try:
        return upscaler.upscale(img, scale=scale, outscale=outscale)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    img = _decode_image(await file.read())
    out, meta = _run(img, scale, outscale)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    headers = {
        "X-Native-Scale": str(meta["native_scale"]),
        "X-Effective-Scale": str(meta["effective_scale"]),
        "X-Output-Size": f"{meta['output_size'][0]}x{meta['output_size'][1]}",
    }
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


class UpscaleRequest(BaseModel):
    image: str  # base64 PNG/JPEG; a `data:image/...;base64,` prefix is tolerated
    scale: int = DEFAULT_SCALE
    outscale: float | None = None
    response_format: str = "b64_json"  # only b64_json is supported (no object storage)


@app.post("/upscale/json")
def do_upscale_json(req: UpscaleRequest) -> dict:
    """JSON-in/JSON-out twin of `/upscale`.

    Mirrors the generation services' OpenAI-shaped `{"data": [{"b64_json": ...}]}`
    envelope so generate -> upscale is a two-call JSON pipeline instead of a
    base64-decode-to-file-then-re-upload dance. Multipart `/upscale` stays the
    primary route: it avoids the ~33% base64 tax on both legs.
    """
    if req.response_format != "b64_json":
        raise HTTPException(status_code=400, detail="only response_format='b64_json' is supported")

    payload = req.image.split(",", 1)[-1] if req.image.startswith("data:") else req.image
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"image is not valid base64: {exc}") from exc

    out, meta = _run(_decode_image(raw), req.scale, req.outscale)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {
        "created": int(time.time()),
        "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}],
        "native_scale": meta["native_scale"],
        "effective_scale": meta["effective_scale"],
        "input_size": meta["input_size"],
        "output_size": meta["output_size"],
    }
