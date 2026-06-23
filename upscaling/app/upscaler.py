"""Spandrel-backed image upscaler with selectable scale.

Spandrel is the model-loading library ComfyUI uses: it auto-detects the
architecture from a bare .pth (ESRGAN / Real-ESRGAN / SwinIR / DAT / ...) and
hands back a ready-to-run model. We deliberately avoid the legacy
`realesrgan`/`basicsr` packages, which carry torchvision-deprecation rot that
breaks on modern (torch 2.7 / Blackwell) builds.

Two native Real-ESRGAN checkpoints are wired in: x2 and x4. Requests pick a
native `scale` (2 or 4); an optional `outscale` float resizes the result to an
arbitrary factor (Lanczos) on top of the nearest native model.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")

# Official first-party Real-ESRGAN release weights, keyed by native scale.
NATIVE_WEIGHTS = {
    2: (
        "RealESRGAN_x2plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ),
    4: (
        "RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
}

_models: dict[int, ImageModelDescriptor] = {}
_lock = threading.Lock()


def _ensure_weights(path: str, url: str) -> None:
    if os.path.exists(path):
        return
    import urllib.request

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted release URL)
    os.replace(tmp, path)


def get_model(scale: int) -> ImageModelDescriptor:
    """Lazily download + load the native model for `scale` (2 or 4), cached."""
    if scale not in NATIVE_WEIGHTS:
        raise ValueError(f"unsupported native scale {scale}; choose one of {sorted(NATIVE_WEIGHTS)}")
    with _lock:
        if scale not in _models:
            fname, url = NATIVE_WEIGHTS[scale]
            path = os.path.join(WEIGHTS_DIR, fname)
            _ensure_weights(path, url)
            model = ModelLoader().load_from_file(path)
            if not isinstance(model, ImageModelDescriptor):
                raise ValueError(f"{path} is not a single-image upscaling model")
            model.cuda().eval()
            _models[scale] = model
    return _models[scale]


def preload(scales: list[int]) -> None:
    for s in scales:
        get_model(s)


@torch.inference_mode()
def _run_native(model: ImageModelDescriptor, img: Image.Image) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()  # 1,C,H,W
    out = model(t)
    out = out.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    return Image.fromarray((out * 255.0).round().astype(np.uint8))


def upscale(img: Image.Image, scale: int = 4, outscale: float | None = None) -> tuple[Image.Image, dict]:
    """Upscale `img`.

    - `scale`: native model to run (2 or 4).
    - `outscale`: if set, final size = round(original * outscale). The nearest
      native model that meets/exceeds it is run, then Lanczos resampled to the
      exact factor. Lets you ask for ×1.5, ×3, ×8, etc.
    """
    src_w, src_h = img.size

    if outscale is not None:
        if outscale <= 0:
            raise ValueError("outscale must be > 0")
        # Pick the smallest native scale that reaches the target, else the largest.
        native = next((s for s in sorted(NATIVE_WEIGHTS) if s >= outscale), max(NATIVE_WEIGHTS))
    else:
        native = scale

    out = _run_native(get_model(native), img)

    target = outscale if outscale is not None else native
    final_w, final_h = round(src_w * target), round(src_h * target)
    if (out.width, out.height) != (final_w, final_h):
        out = out.resize((final_w, final_h), Image.LANCZOS)

    meta = {
        "native_scale": native,
        "effective_scale": round(out.width / src_w, 4),
        "input_size": [src_w, src_h],
        "output_size": [out.width, out.height],
    }
    return out, meta
