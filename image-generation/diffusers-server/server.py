"""Reusable OpenAI-compatible image server for any 🤗 diffusers text-to-image pipeline.

One image, many models: the concrete pipeline is chosen at runtime via env vars, so
the same container serves Z-Image-Turbo, Qwen-Image, etc. by swapping MODEL_ID /
PIPELINE_CLASS. Exposes the OpenAI Images shape (`POST /v1/images/generations`) so it
drops straight into LibreChat, a Gradio app, or any OpenAI client.

Env config:
  MODEL_ID            HF repo id (e.g. Tongyi-MAI/Z-Image-Turbo)            [required]
  PIPELINE_CLASS      diffusers class name (e.g. ZImagePipeline)            [required]
  SERVED_MODEL_NAME   name reported by /v1/models and matched on requests   [= MODEL_ID]
  TORCH_DTYPE         bfloat16 | float16 | float32                          [bfloat16]
  DEFAULT_STEPS       num_inference_steps when caller omits it              [9]
  DEFAULT_GUIDANCE    guidance value when caller omits it                   [0.0]
  GUIDANCE_PARAM      pipeline kwarg for guidance (guidance_scale or        [guidance_scale]
                      true_cfg_scale — Qwen-Image needs true_cfg_scale)
  DEFAULT_NEG_PROMPT  negative_prompt used when caller omits it             [""]
                      (Qwen-Image needs at least " " to enable CFG)
  DEFAULT_SIZE        WxH used when caller omits size                       [1024x1024]
  LOW_CPU_MEM_USAGE   "1"/"0" for from_pretrained low_cpu_mem_usage         [1]
  DEVICE_MAP          accelerate device_map (e.g. "cuda") — streams weights [unset]
                      straight to GPU, avoiding full CPU-RAM materialization
                      (required for big models on low-RAM hosts). When set,
                      .to("cuda") is skipped.
  ENABLE_CPU_OFFLOAD  "1" to enable model CPU offload (low-VRAM)            [0]
  ATTENTION_BACKEND   optional transformer attention backend (e.g. flash)  [unset]
  TRUST_REMOTE_CODE   "1" to pass trust_remote_code=True to from_pretrained [0]
  PORT                listen port                                           [8000]
"""
import base64
import io
import os
import time
from importlib import import_module
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_ID = os.environ["MODEL_ID"]
PIPELINE_CLASS = os.environ["PIPELINE_CLASS"]
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL_ID)
TORCH_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}[os.environ.get("TORCH_DTYPE", "bfloat16")]
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "9"))
DEFAULT_GUIDANCE = float(os.environ.get("DEFAULT_GUIDANCE", "0.0"))
GUIDANCE_PARAM = os.environ.get("GUIDANCE_PARAM", "guidance_scale")
DEFAULT_NEG_PROMPT = os.environ.get("DEFAULT_NEG_PROMPT", "")
DEFAULT_SIZE = os.environ.get("DEFAULT_SIZE", "1024x1024")
LOW_CPU_MEM_USAGE = os.environ.get("LOW_CPU_MEM_USAGE", "1") == "1"
DEVICE_MAP = os.environ.get("DEVICE_MAP", "").strip()
ENABLE_CPU_OFFLOAD = os.environ.get("ENABLE_CPU_OFFLOAD", "0") == "1"
ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "").strip()
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "0") == "1"
PORT = int(os.environ.get("PORT", "8000"))

pipe = None  # populated on startup


def _parse_size(size: str) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in size.lower().split("x"))
        return w, h
    except Exception:
        raise HTTPException(400, f"invalid size {size!r}, expected e.g. '1024x1024'")


app = FastAPI(title="diffusers-openai-image-server")


@app.on_event("startup")
def _load():
    global pipe
    cls = getattr(import_module("diffusers"), PIPELINE_CLASS)
    print(f"[server] loading {MODEL_ID} as {PIPELINE_CLASS} ({TORCH_DTYPE}) ...", flush=True)
    t0 = time.time()
    kwargs = {"torch_dtype": TORCH_DTYPE, "low_cpu_mem_usage": LOW_CPU_MEM_USAGE}
    if TRUST_REMOTE_CODE:
        kwargs["trust_remote_code"] = True
    if DEVICE_MAP:
        # stream weights straight to the GPU; peak CPU RAM ≈ one shard
        kwargs["device_map"] = DEVICE_MAP
    pipe = cls.from_pretrained(MODEL_ID, **kwargs)
    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    elif not DEVICE_MAP:
        pipe.to("cuda")
    if ATTENTION_BACKEND:
        try:
            pipe.transformer.set_attention_backend(ATTENTION_BACKEND)
            print(f"[server] attention backend: {ATTENTION_BACKEND}", flush=True)
        except Exception as e:  # non-fatal: fall back to default attention
            print(f"[server] WARN could not set attention backend: {e}", flush=True)
    print(f"[server] ready in {time.time() - t0:.1f}s — serving '{SERVED_MODEL_NAME}'", flush=True)


class ImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: int = 1
    size: Optional[str] = None
    response_format: str = "b64_json"  # only b64_json is supported (no object storage)
    # non-OpenAI extras (accepted as plain body fields):
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok" if pipe is not None else "loading", "model": SERVED_MODEL_NAME}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "local"}]}


@app.get("/v1/config")
def config():
    """Per-model generation defaults, so UIs can adapt sliders to the model."""
    return {
        "model": SERVED_MODEL_NAME,
        "default_steps": DEFAULT_STEPS,
        "default_guidance": DEFAULT_GUIDANCE,
        "guidance_param": GUIDANCE_PARAM,
        "default_size": DEFAULT_SIZE,
    }


@app.post("/v1/images/generations")
def generate(req: ImageRequest):
    if pipe is None:
        raise HTTPException(503, "model still loading")
    if req.response_format != "b64_json":
        raise HTTPException(400, "only response_format='b64_json' is supported")
    w, h = _parse_size(req.size or DEFAULT_SIZE)
    steps = req.num_inference_steps if req.num_inference_steps is not None else DEFAULT_STEPS
    guidance = req.guidance_scale if req.guidance_scale is not None else DEFAULT_GUIDANCE

    call_kwargs = {
        "prompt": req.prompt,
        "height": h,
        "width": w,
        "num_inference_steps": steps,
        GUIDANCE_PARAM: guidance,
        "num_images_per_prompt": req.n,
    }
    neg = req.negative_prompt if req.negative_prompt is not None else DEFAULT_NEG_PROMPT
    if neg:
        call_kwargs["negative_prompt"] = neg
    if req.seed is not None:
        call_kwargs["generator"] = torch.Generator("cuda").manual_seed(req.seed)

    t0 = time.time()
    images = pipe(**call_kwargs).images
    dt = time.time() - t0

    data = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})
    print(f"[server] {req.n}x {w}x{h} steps={steps} cfg={guidance} -> {dt:.1f}s", flush=True)
    return {"created": int(t0), "data": data}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
