"""SUPIR (CVPR 2024) diffusion restoration, wrapped for one-call serving.

SUPIR is a *restoration* model, not an interpolator: it applies SDXL's generative
prior to invent plausible detail. That makes it the opposite trade-off from the
Real-ESRGAN service next door — see the perception-distortion note in README.md.

Upstream drives this from test.py + argparse + a hardcoded-path YAML. Here the
config is loaded and re-pointed at in-container weight paths in memory, so the
vendored checkout under /app/supir_src stays untouched.

LLaVA is deliberately not loaded. Upstream uses a 13B captioner to auto-write the
restoration prompt; this service takes `prompt` over the API instead, which drops
~14 GB of weights and a large chunk of VRAM for the same knob.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time

import torch
from omegaconf import OmegaConf
from PIL import Image

SUPIR_SRC = os.environ.get("SUPIR_SRC", "/app/supir_src")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
SIGN = os.environ.get("SUPIR_SIGN", "Q")  # Q = high generalization, F = light degradation
DEVICE = os.environ.get("SUPIR_DEVICE", "cuda")

# Upstream's defaults (test.py), which the API layer exposes as overridable fields.
DEFAULT_POSITIVE = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, "
    "extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect "
    "without deformations."
)
DEFAULT_NEGATIVE = (
    "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
    "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low "
    "quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
)

CKPTS = {
    "SUPIR_CKPT_Q": os.path.join(WEIGHTS_DIR, "SUPIR-v0Q_fp16.safetensors"),
    "SUPIR_CKPT_F": os.path.join(WEIGHTS_DIR, "SUPIR-v0F_fp16.safetensors"),
    "SDXL_CKPT": os.path.join(WEIGHTS_DIR, "sd_xl_base_1.0_0.9vae.safetensors"),
}

_model = None
_lock = threading.Lock()
_load_error: str | None = None


def _runtime_config() -> str:
    """Rewrite upstream's YAML with in-container paths; return a temp config path."""
    base = os.path.join(SUPIR_SRC, "options", "SUPIR_v0.yaml")
    cfg = OmegaConf.load(base)
    cfg.SDXL_CKPT = CKPTS["SDXL_CKPT"]
    cfg.SUPIR_CKPT_Q = CKPTS["SUPIR_CKPT_Q"]
    cfg.SUPIR_CKPT_F = CKPTS["SUPIR_CKPT_F"] if os.path.exists(CKPTS["SUPIR_CKPT_F"]) else None
    cfg.SUPIR_CKPT = None  # only the sign-selected delta is applied
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="supir_runtime_")
    os.close(fd)
    OmegaConf.save(cfg, path)
    return path


def missing_weights() -> list[str]:
    need = ["SDXL_CKPT", f"SUPIR_CKPT_{SIGN}"]
    missing = [CKPTS[k] for k in need if not os.path.exists(CKPTS[k])]
    for extra in (
        os.path.join(WEIGHTS_DIR, "clip-vit-large-patch14", "config.json"),
        os.path.join(
            WEIGHTS_DIR, "CLIP-ViT-bigG-14-laion2B-39B-b160k", "open_clip_pytorch_model.bin"
        ),
    ):
        if not os.path.exists(extra):
            missing.append(extra)
    return missing


def load() -> None:
    """Build the model on GPU. Slow (tens of seconds) and ~20 GB of state dicts."""
    global _model, _load_error
    with _lock:
        if _model is not None:
            return
        missing = missing_weights()
        if missing:
            _load_error = f"missing checkpoints: {missing}"
            raise RuntimeError(_load_error)

        from SUPIR.util import create_SUPIR_model  # noqa: PLC0415 (needs PYTHONPATH)

        cfg_path = _runtime_config()
        try:
            t0 = time.time()
            model = create_SUPIR_model(cfg_path, SUPIR_sign=SIGN)
            if os.environ.get("SUPIR_HALF_PARAMS", "1") == "1":
                model = model.half()
            if os.environ.get("SUPIR_TILE_VAE", "1") == "1":
                model.init_tile_vae(
                    encoder_tile_size=int(os.environ.get("SUPIR_ENC_TILE", "512")),
                    decoder_tile_size=int(os.environ.get("SUPIR_DEC_TILE", "64")),
                )
            model.ae_dtype = torch.bfloat16
            model.model.dtype = torch.float16
            _model = model.to(DEVICE)
            print(f"[supir] loaded sign={SIGN} in {time.time() - t0:.1f}s", flush=True)
        finally:
            os.unlink(cfg_path)


def is_loaded() -> bool:
    return _model is not None


def load_error() -> str | None:
    return _load_error


@torch.no_grad()
def restore(
    img: Image.Image,
    *,
    upscale: float = 2.0,
    prompt: str = "",
    positive_prompt: str = DEFAULT_POSITIVE,
    negative_prompt: str = DEFAULT_NEGATIVE,
    edm_steps: int = 50,
    s_cfg: float = 4.0,
    s_stage1: float = -1.0,
    s_stage2: float = 1.0,
    s_churn: float = 5.0,
    s_noise: float = 1.01,
    seed: int = 1234,
    color_fix_type: str = "Wavelet",
    min_size: int = 1024,
) -> tuple[Image.Image, dict]:
    """Restore + upscale `img`. Returns (image, metadata)."""
    if _model is None:
        raise RuntimeError("model not loaded")
    if color_fix_type not in ("Wavelet", "AdaIn", "None"):
        raise ValueError("color_fix_type must be one of Wavelet, AdaIn, None")
    if upscale <= 0:
        raise ValueError("upscale must be > 0")

    from SUPIR.util import PIL2Tensor, Tensor2PIL  # noqa: PLC0415

    src_w, src_h = img.size
    t0 = time.time()

    # PIL2Tensor also enforces min_size and rounds to a /64 grid; h0,w0 are the
    # target dims to crop back to after sampling.
    lq, h0, w0 = PIL2Tensor(img.convert("RGB"), upsacle=upscale, min_size=min_size)
    lq = lq.unsqueeze(0).to(DEVICE)[:, :3, :, :]

    samples = _model.batchify_sample(
        lq,
        [prompt],
        num_steps=edm_steps,
        restoration_scale=s_stage1,
        s_churn=s_churn,
        s_noise=s_noise,
        cfg_scale=s_cfg,
        control_scale=s_stage2,
        seed=seed,
        num_samples=1,
        p_p=positive_prompt,
        n_p=negative_prompt,
        color_fix_type=color_fix_type,
        use_linear_CFG=True,
        use_linear_control_scale=False,
        cfg_scale_start=1.0,
        control_scale_start=0.0,
    )
    out = Tensor2PIL(samples[0], h0, w0)
    dt = time.time() - t0

    meta = {
        "sign": SIGN,
        "requested_upscale": upscale,
        "effective_scale": round(out.width / src_w, 4),
        "input_size": [src_w, src_h],
        "output_size": [out.width, out.height],
        "edm_steps": edm_steps,
        "seed": seed,
        "duration_s": round(dt, 2),
    }
    return out, meta
