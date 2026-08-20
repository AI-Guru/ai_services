#!/usr/bin/env python3
"""
Stable Audio 3 Small-SFX as a resident FastAPI service.

Text-to-SFX, audio-to-audio restyling, and inpainting/continuation over one 433M
latent-diffusion model held on the GPU for the process lifetime.

  POST /v1/audio/sync    -> the rendered clip in the response body
  POST /v1/audio         -> {"id": ...}, poll /v1/audio/{id}, fetch /v1/audio/{id}/content
  GET  /health           -> residency + VRAM
  GET  /ui               -> Gradio front-end, same process, same model

WHY A GENERATION LOCK ON A 2 GB MODEL — the card is shared. Small-SFX peaks around
1.7-2.4 GB, so unlike the video stacks in this repo it can sit beside a resident LLM.
The lock is not about VRAM, it is about not interleaving two denoising loops through the
same weights and thrashing the SM scheduler; `batch_size` is the supported way to get
several clips at once and is strictly faster than concurrent single requests.

TWO UPSTREAM TRAPS ARE HANDLED HERE, both documented in README.md:
  * init_audio/inpaint_audio want (sample_rate, tensor). The upstream examples pass
    torchaudio.load() straight in, which yields (tensor, sample_rate) — reversed.
  * cfg_scale and negative_prompt are inert on the post-trained checkpoints. Small-SFX
    is post-trained, so this server reports them back as ignored rather than pretending.
"""
import asyncio, base64, io, json, os, time, traceback, uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("SA3_MODEL", "small-sfx")
OUTPUTS = os.environ.get("SA3_OUTPUTS", "/outputs")
INPUTS = os.environ.get("SA3_INPUTS", "/inputs")
PORT = int(os.environ.get("SA3_PORT", "8400"))
HALF = os.environ.get("SA3_HALF", "1") == "1"
# Upstream defaults chunked decoding ON for every model. It exists to cap peak VRAM on
# small cards; here it costs compute and can leave stitching artefacts at chunk seams,
# and the ceiling it protects is ~2.4 GB on a 96 GB card. Off by default, overridable.
CHUNKED = os.environ.get("SA3_CHUNKED_DECODE", "0") == "1"

# Post-trained checkpoints ignore CFG entirely; only the '-base' ones respond to it.
POST_TRAINED = not MODEL_ID.endswith("-base")
MAX_DURATION = 120.0  # SAME-S context limit for both Small checkpoints

STATE = {"model": None, "sr": None, "loaded_at": None, "gen_lock": None, "formats": []}
JOBS = {}


def load_audio_field(value, name):
    """Accept a filename under /inputs, a data: URI, or raw base64, and return the
    (sample_rate, tensor) tuple stable_audio_3 actually wants — note the order."""
    import torchaudio
    if not value:
        return None
    path = os.path.join(INPUTS, value) if isinstance(value, str) else None
    if path and os.path.exists(path):
        wav, sr = torchaudio.load(path)
        return (sr, wav)
    if isinstance(value, str) and value.startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        wav, sr = torchaudio.load(io.BytesIO(base64.b64decode(value)))
        return (sr, wav)
    except Exception as e:
        raise HTTPException(
            400, f"{name}: not a file under {INPUTS} nor decodable audio ({type(e).__name__}: {e})")


def load_model():
    from stable_audio_3 import StableAudioModel
    import soundfile as sf

    t0 = time.time()
    print(f"loading Stable Audio 3 '{MODEL_ID}' ...", flush=True)

    # Both source repos are gated. from_pretrained resolves them through huggingface_hub,
    # which reads HF_TOKEN from the environment; a 401 here means the token is missing or
    # the licence has not been accepted on the Hub. See README "Two gates, not one".
    model = StableAudioModel.from_pretrained(MODEL_ID, device="cuda", model_half=HALF)

    sr = model.model.sample_rate
    STATE["formats"] = sorted(k.lower() for k in sf.available_formats())

    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"ready in {time.time()-t0:.0f}s — {alloc:.2f} GB resident of {total:.0f} GB, "
          f"{sr} Hz, half={HALF}, post_trained={POST_TRAINED}", flush=True)
    return model, sr


@asynccontextmanager
async def lifespan(app):
    os.makedirs(OUTPUTS, exist_ok=True)
    os.makedirs(INPUTS, exist_ok=True)
    STATE["gen_lock"] = asyncio.Lock()
    STATE["model"], STATE["sr"] = load_model()
    STATE["loaded_at"] = time.time()
    yield
    STATE["model"] = None


app = FastAPI(title=f"Stable Audio 3 {MODEL_ID}", lifespan=lifespan)


class SfxRequest(BaseModel):
    prompt: str | list[str]
    duration: float | list[float] = 7.0
    steps: int = 8
    seed: int = -1
    batch_size: int = 1
    format: str = Field("wav", description="wav | flac | ogg | mp3, subject to libsndfile")
    prefix: str = "sfx"
    chunked_decode: bool | None = None

    # Base-checkpoint-only. Accepted and echoed back as ignored on post-trained weights.
    negative_prompt: str | list[str] | None = None
    cfg_scale: float = 1.0

    # Audio-to-audio: noise the source, then denoise toward the prompt.
    init_audio: str | None = Field(None, description="filename under /inputs, data URI or base64")
    init_noise_level: float = Field(1.0, ge=0.0, le=1.0)

    # Inpainting / continuation. Floats for one region, equal-length lists for several.
    inpaint_audio: str | None = None
    inpaint_mask_start_seconds: float | list[float] | None = None
    inpaint_mask_end_seconds: float | list[float] | None = None


def _generate(req: SfxRequest):
    import soundfile as sf

    model = STATE["model"]
    if model is None:
        raise HTTPException(503, "model not loaded")

    fmt = req.format.lower().lstrip(".")
    if fmt not in STATE["formats"]:
        raise HTTPException(
            400, f"format '{fmt}' unsupported by this libsndfile build; have: {STATE['formats']}")

    longest = max(req.duration) if isinstance(req.duration, list) else req.duration
    if longest > MAX_DURATION:
        raise HTTPException(
            400, f"duration {longest}s exceeds the {MAX_DURATION:.0f}s ceiling for {MODEL_ID}")

    kwargs = dict(
        prompt=req.prompt,
        duration=req.duration,
        steps=req.steps,
        seed=req.seed,
        batch_size=req.batch_size,
        cfg_scale=req.cfg_scale,
        chunked_decode=CHUNKED if req.chunked_decode is None else req.chunked_decode,
    )
    if req.negative_prompt:
        kwargs["negative_prompt"] = req.negative_prompt

    init = load_audio_field(req.init_audio, "init_audio")
    if init is not None:
        kwargs["init_audio"] = init
        kwargs["init_noise_level"] = req.init_noise_level

    inpaint = load_audio_field(req.inpaint_audio, "inpaint_audio")
    if inpaint is not None:
        if req.inpaint_mask_start_seconds is None or req.inpaint_mask_end_seconds is None:
            raise HTTPException(400, "inpaint_audio needs both mask start and end seconds")
        starts, ends = req.inpaint_mask_start_seconds, req.inpaint_mask_end_seconds
        if isinstance(starts, list) != isinstance(ends, list) or (
                isinstance(starts, list) and len(starts) != len(ends)):
            raise HTTPException(400, "inpaint mask start/end must both be scalars or equal-length lists")
        kwargs["inpaint_audio"] = inpaint
        kwargs["inpaint_mask_start_seconds"] = starts
        kwargs["inpaint_mask_end_seconds"] = ends

    mode = ("inpaint" if inpaint is not None else
            "audio2audio" if init is not None else "text2audio")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    audio = model.generate(**kwargs)          # [batch, channels, samples], float32 in [-1, 1]
    # generate() returns as soon as the kernels are QUEUED. Without this sync the timer
    # stops mid-flight and the remaining GPU work gets billed to the .cpu() in the file
    # write below — which showed up as a suspiciously flat 0.13 s for every duration from
    # 5 s to 120 s. Sync first, then measure.
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    sr = STATE["sr"]
    stamp = f"{req.prefix}_{int(time.time())}"
    files = []
    for i in range(audio.shape[0]):
        name = f"{stamp}_{i}.{fmt}" if audio.shape[0] > 1 else f"{stamp}.{fmt}"
        path = os.path.join(OUTPUTS, name)
        # soundfile wants [samples, channels]; the model emits [channels, samples].
        sf.write(path, audio[i].transpose(0, 1).cpu().numpy(), sr, format=fmt.upper())
        files.append({"file": name, "path": path})

    rendered = audio.shape[-1] / sr
    info = {
        "path": files[0]["path"],
        "file": files[0]["file"],
        "files": files,
        "model": MODEL_ID,
        "mode": mode,
        "sample_rate": sr,
        "channels": int(audio.shape[1]),
        "duration_s": round(rendered, 2),
        "steps": req.steps,
        "seed": req.seed,
        "batch_size": req.batch_size,
        "format": fmt,
        "generation_s": round(elapsed, 2),
        "realtime_factor": round(rendered * audio.shape[0] / max(elapsed, 1e-6), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    if POST_TRAINED and (req.cfg_scale != 1.0 or req.negative_prompt):
        info["ignored"] = ("cfg_scale and negative_prompt have no effect on the post-trained "
                           f"'{MODEL_ID}' checkpoint — use '{MODEL_ID}-base' for guided sampling")
    return info


@app.get("/health")
def health():
    m = STATE["model"]
    out = {"status": "ok" if m is not None else "loading",
           "resident": m is not None,
           "model": MODEL_ID,
           "post_trained": POST_TRAINED,
           "sample_rate": STATE["sr"],
           "max_duration_s": MAX_DURATION,
           "half": HALF,
           "chunked_decode_default": CHUNKED,
           "formats": STATE["formats"]}
    if torch.cuda.is_available():
        out["vram"] = {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        }
    if STATE["loaded_at"]:
        out["uptime_s"] = round(time.time() - STATE["loaded_at"])
    return out


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": f"stable-audio-3-{MODEL_ID}", "object": "model"}]}


_MEDIA = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg", "mp3": "audio/mpeg"}


@app.post("/v1/audio/sync")
async def audio_sync(req: SfxRequest):
    async with STATE["gen_lock"]:
        try:
            info = await asyncio.to_thread(_generate, req)
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"{type(e).__name__}: {e}")
    return FileResponse(info["path"],
                        media_type=_MEDIA.get(info["format"], "application/octet-stream"),
                        filename=info["file"],
                        headers={"X-SA3-Info": json.dumps(info)})


@app.post("/v1/audio")
async def audio_async(req: SfxRequest):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"id": job_id, "status": "queued", "created": time.time()}

    async def run():
        async with STATE["gen_lock"]:
            JOBS[job_id]["status"] = "running"
            try:
                JOBS[job_id].update(await asyncio.to_thread(_generate, req), status="completed")
            except Exception as e:
                traceback.print_exc()
                JOBS[job_id].update(status="failed", error=f"{type(e).__name__}: {e}")

    asyncio.create_task(run())
    return JSONResponse({"id": job_id, "status": "queued"}, status_code=202)


@app.get("/v1/audio/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")
    return JOBS[job_id]


@app.get("/v1/audio/{job_id}/content")
def job_content(job_id: str, index: int = 0):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.get("status") != "completed":
        raise HTTPException(409, f"job is {job.get('status')}")
    files = job["files"]
    if not 0 <= index < len(files):
        raise HTTPException(404, f"index {index} out of range (batch of {len(files)})")
    return FileResponse(files[index]["path"],
                        media_type=_MEDIA.get(job["format"], "application/octet-stream"),
                        filename=files[index]["file"])


# ---------------------------------------------------------------------------------------
# Gradio UI, mounted on this same app so it shares the loaded model and the gen lock.
# ui.py is handed everything it needs as arguments rather than importing this module: the
# server runs as __main__, so `import server` inside ui.py would execute this file a
# second time and give the UI its own STATE with no model in it.
#
# A UI failure must not take the REST API down with it.
# ---------------------------------------------------------------------------------------
try:
    from fastapi.responses import RedirectResponse
    from ui import mount_ui

    mount_ui(app, state=STATE, generate=_generate, request_cls=SfxRequest,
             inputs_dir=INPUTS, outputs_dir=OUTPUTS, model_id=MODEL_ID,
             post_trained=POST_TRAINED, max_duration=MAX_DURATION)

    @app.get("/", include_in_schema=False)
    def _root():
        return RedirectResponse("/ui")
except Exception as e:
    traceback.print_exc()
    print(f"UI NOT mounted ({type(e).__name__}: {e}) — REST API is unaffected", flush=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
