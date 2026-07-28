"""FastAPI wrapper for SeedVR2-3B.

Route shape mirrors the two upscalers in upscaling/ (POST /upscale multipart,
POST /upscale/json b64, GET /health) so all three are interchangeable by base URL
and can be compared by the same harness (upscaling/compare_upscalers.py).

The knobs differ because the model does. SeedVR2 takes no prompt — its text
conditioning is precomputed and frozen — and is one-step by design, so there is no
step count worth exposing beyond an override for experimentation.
"""
from __future__ import annotations

import base64
import binascii
import io
import os
import threading
import time

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

from . import seedvr2_engine as engine

app = FastAPI(title="ai_services SeedVR2 restorer", version="0.1.0")

DEFAULT_SCALE = float(os.environ.get("SEEDVR2_DEFAULT_SCALE", "4"))
DEFAULT_STEPS = int(os.environ.get("SEEDVR2_DEFAULT_STEPS", "1"))
DEFAULT_CFG = float(os.environ.get("SEEDVR2_DEFAULT_CFG", "1.0"))
MAX_PIXELS = int(os.environ.get("SEEDVR2_MAX_OUTPUT_PIXELS", str(4096 * 4096)))

_boot_error: str | None = None


@app.on_event("startup")
def _startup() -> None:
    def _bg() -> None:
        global _boot_error
        try:
            engine.load()
        except Exception as exc:  # noqa: BLE001
            _boot_error = str(exc)
            print(f"[seedvr2] LOAD FAILED: {exc}", flush=True)
            import traceback

            traceback.print_exc()

    threading.Thread(target=_bg, daemon=True).start()


@app.get("/health")
def health() -> JSONResponse:
    """200 only when the model is usable; 503 while loading or after a failure.

    Not a flat 200 — the compose healthcheck is `curl -fsS`, and returning 200 with
    status='error' would mark a container healthy whose model never loaded, the
    false-healthy trap the repo CLAUDE.md warns about.
    """
    vram = {}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        vram = {
            "gpu_used_gb": round((total - free) / 1e9, 2),
            "gpu_total_gb": round(total / 1e9, 2),
        }
    status = "ok" if engine.is_loaded() else ("error" if _boot_error else "loading")
    body = {
        "status": status,
        "model": "SeedVR2-3B",
        "error": _boot_error,
        "missing_weights": engine.missing_weights(),
        "default_scale": DEFAULT_SCALE,
        "default_steps": DEFAULT_STEPS,
        "prompt_supported": False,  # conditioning is precomputed, see engine docstring
        **vram,
    }
    return JSONResponse(body, status_code=200 if status == "ok" else 503)


def _decode_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad image: {exc}") from exc
    return img


def _run(img: Image.Image, params: dict) -> tuple[Image.Image, dict]:
    if _boot_error:
        raise HTTPException(status_code=503, detail=f"model failed to load: {_boot_error}")
    if not engine.is_loaded():
        raise HTTPException(status_code=503, detail="model still loading")

    scale = params.get("scale") or DEFAULT_SCALE
    px = int(img.width * scale) * int(img.height * scale)
    if px > MAX_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"requested output {px} px exceeds SEEDVR2_MAX_OUTPUT_PIXELS={MAX_PIXELS}",
        )
    try:
        return engine.restore(img, **params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail=f"CUDA OOM: {exc}") from exc


@app.post("/upscale")
async def do_upscale(
    file: UploadFile = File(...),
    scale: float = Form(DEFAULT_SCALE),
    sample_steps: int = Form(DEFAULT_STEPS),
    cfg_scale: float = Form(DEFAULT_CFG),
    seed: int = Form(666),
) -> Response:
    """Restore + upscale the uploaded image. No prompt: see /health.prompt_supported."""
    img = _decode_image(await file.read())
    out, meta = _run(
        img, dict(scale=scale, sample_steps=sample_steps, cfg_scale=cfg_scale, seed=seed)
    )
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Effective-Scale": str(meta["effective_scale"]),
            "X-Output-Size": f"{meta['output_size'][0]}x{meta['output_size'][1]}",
            "X-Duration-S": str(meta["duration_s"]),
            "X-Model": "SeedVR2-3B",
        },
    )


class RestoreRequest(BaseModel):
    image: str  # base64; a `data:image/...;base64,` prefix is tolerated
    scale: float = DEFAULT_SCALE
    sample_steps: int = DEFAULT_STEPS
    cfg_scale: float = DEFAULT_CFG
    seed: int = 666
    response_format: str = "b64_json"


@app.post("/upscale/json")
def do_upscale_json(req: RestoreRequest) -> dict:
    """JSON twin of /upscale — same envelope as the sibling services."""
    if req.response_format != "b64_json":
        raise HTTPException(status_code=400, detail="only response_format='b64_json' is supported")

    payload = req.image.split(",", 1)[-1] if req.image.startswith("data:") else req.image
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"image is not valid base64: {exc}") from exc

    out, meta = _run(
        _decode_image(raw),
        dict(
            scale=req.scale,
            sample_steps=req.sample_steps,
            cfg_scale=req.cfg_scale,
            seed=req.seed,
        ),
    )
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {
        "created": int(time.time()),
        "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}],
        **meta,
    }
