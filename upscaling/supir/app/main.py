"""FastAPI wrapper for SUPIR — the 'creative' tier next to the Real-ESRGAN service.

Route shape deliberately mirrors ../app/main.py (POST /upscale multipart, POST
/upscale/json b64, GET /health) so a client can switch base URL and keep working.
The knobs differ, because the models differ: Real-ESRGAN takes a scale factor and
nothing else; SUPIR takes a prompt, a step count and a guidance scale, because it
is a conditional diffusion model.

Timing note: SUPIR is ~50 diffusion steps on SDXL, so a call is tens of seconds,
not the sibling's 0.85 s. These routes are synchronous — set generous client
timeouts. See README.md for why an async job API is the honest long-term shape.
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

from . import supir_engine as engine

app = FastAPI(title="ai_services SUPIR restorer", version="0.1.0")

DEFAULT_UPSCALE = float(os.environ.get("SUPIR_DEFAULT_UPSCALE", "2"))
DEFAULT_STEPS = int(os.environ.get("SUPIR_DEFAULT_STEPS", "50"))
DEFAULT_CFG = float(os.environ.get("SUPIR_DEFAULT_CFG", "4.0"))
# Guard against a request that would OOM the card or run for many minutes.
MAX_PIXELS = int(os.environ.get("SUPIR_MAX_OUTPUT_PIXELS", str(4096 * 4096)))

_boot_error: str | None = None


@app.on_event("startup")
def _startup() -> None:
    """Load in a background thread so /health answers 'loading' instead of hanging."""

    def _bg() -> None:
        global _boot_error
        try:
            engine.load()
        except Exception as exc:  # noqa: BLE001
            _boot_error = str(exc)
            print(f"[supir] LOAD FAILED: {exc}", flush=True)

    threading.Thread(target=_bg, daemon=True).start()


@app.get("/health")
def health() -> JSONResponse:
    """200 only when the model is actually usable.

    Deliberately NOT a flat 200: the compose healthcheck is `curl -fsS`, so
    returning 200 with status='error' would mark a container healthy whose model
    failed to load — the false-healthy trap called out in the repo CLAUDE.md.
    Loading answers 503 too, so `up --wait` blocks until the weights are really in.
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
        "model": f"SUPIR-v0{engine.SIGN}",
        "error": _boot_error,
        "missing_weights": engine.missing_weights(),
        "default_upscale": DEFAULT_UPSCALE,
        "default_steps": DEFAULT_STEPS,
        **vram,
    }
    return JSONResponse(body, status_code=200 if status == "ok" else 503)


def _require_ready() -> None:
    if _boot_error:
        raise HTTPException(status_code=503, detail=f"model failed to load: {_boot_error}")
    if not engine.is_loaded():
        raise HTTPException(status_code=503, detail="model still loading")


def _decode_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad image: {exc}") from exc
    return img


def _check_budget(img: Image.Image, upscale: float) -> None:
    px = int(img.width * upscale) * int(img.height * upscale)
    if px > MAX_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"requested output {int(img.width * upscale)}x{int(img.height * upscale)} "
                f"= {px} px exceeds SUPIR_MAX_OUTPUT_PIXELS={MAX_PIXELS}"
            ),
        )


def _run(img: Image.Image, params: dict) -> tuple[Image.Image, dict]:
    _require_ready()
    _check_budget(img, params.get("upscale", DEFAULT_UPSCALE))
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
    upscale: float = Form(DEFAULT_UPSCALE),
    prompt: str = Form(""),
    negative_prompt: str = Form(engine.DEFAULT_NEGATIVE),
    edm_steps: int = Form(DEFAULT_STEPS),
    s_cfg: float = Form(DEFAULT_CFG),
    s_stage2: float = Form(1.0),
    seed: int = Form(1234),
    color_fix_type: str = Form("Wavelet"),
) -> Response:
    """Restore + upscale the uploaded image.

    `prompt` replaces upstream's LLaVA caption — describing the subject measurably
    improves the result. `s_cfg` trades fidelity (lower) against invention (higher);
    `s_stage2` is the control strength, dropped below 1.0 for heavier degradation.
    """
    img = _decode_image(await file.read())
    out, meta = _run(
        img,
        dict(
            upscale=upscale,
            prompt=prompt,
            negative_prompt=negative_prompt,
            edm_steps=edm_steps,
            s_cfg=s_cfg,
            s_stage2=s_stage2,
            seed=seed,
            color_fix_type=color_fix_type,
        ),
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
            "X-Model": f"SUPIR-v0{meta['sign']}",
        },
    )


class RestoreRequest(BaseModel):
    image: str  # base64; a `data:image/...;base64,` prefix is tolerated
    upscale: float = DEFAULT_UPSCALE
    prompt: str = ""
    negative_prompt: str = engine.DEFAULT_NEGATIVE
    edm_steps: int = DEFAULT_STEPS
    s_cfg: float = DEFAULT_CFG
    s_stage2: float = 1.0
    seed: int = 1234
    color_fix_type: str = "Wavelet"
    response_format: str = "b64_json"


@app.post("/upscale/json")
def do_upscale_json(req: RestoreRequest) -> dict:
    """JSON twin of /upscale, same envelope as the sibling upscaler and the
    generation services: {"created", "data": [{"b64_json"}], ...metadata}."""
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
            upscale=req.upscale,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            edm_steps=req.edm_steps,
            s_cfg=req.s_cfg,
            s_stage2=req.s_stage2,
            seed=req.seed,
            color_fix_type=req.color_fix_type,
        ),
    )
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {
        "created": int(time.time()),
        "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}],
        **meta,
    }
