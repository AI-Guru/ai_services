"""SeedVR2-3B wrapped for one-call serving.

Upstream drives this from projects/inference_seedvr2_3b.py: an argparse CLI that
torchrun-launches, scans a directory of videos, and writes files out. This is that
generation loop refactored for one image, one GPU, in-process — same transforms,
same call sequence, no distributed launcher and no filesystem round-trip.

Notable differences from the two upscalers in upscaling/:

  * SeedVR2 takes a TARGET RESOLUTION, not a scale factor. NaResize maps the input
    onto sqrt(res_h*res_w) with downsample_only=False, because the model was only
    trained at high resolution. `scale` in this module's API is a convenience that
    computes the target from the input size.
  * Conditioning is FROZEN. pos_emb.pt / neg_emb.pt are precomputed text embeddings
    shipped with the weights; there is no text encoder and no prompt to pass.
  * It is one-step (sample_steps=1) by design — that is the paper's contribution.
  * The DiT and VAE are shuttled between GPU and CPU around encode/decode, as
    upstream does, to keep peak VRAM down.
"""
from __future__ import annotations

import os
import threading
import time

# torch.distributed must be initialised before SeedVR's common.distributed helpers
# are imported: init_torch() calls dist.init_process_group unconditionally, and
# get_global_rank()/get_world_size() read these. Single process, single GPU.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", os.environ.get("SEEDVR2_DIST_PORT", "29511"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

SEEDVR_SRC = os.environ.get("SEEDVR_SRC", "/app/seedvr_src")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
DIT_CKPT = os.path.join(WEIGHTS_DIR, os.environ.get("SEEDVR2_DIT", "seedvr2_ema_3b.pth"))
VAE_CKPT = os.path.join(WEIGHTS_DIR, "ema_vae.pth")
POS_EMB = os.path.join(WEIGHTS_DIR, "pos_emb.pt")
NEG_EMB = os.path.join(WEIGHTS_DIR, "neg_emb.pt")
CONFIG = os.environ.get("SEEDVR2_CONFIG", "configs_3b/main.yaml")

_runner = None
_embeds: dict | None = None
_lock = threading.Lock()
_load_error: str | None = None


def missing_weights() -> list[str]:
    return [p for p in (DIT_CKPT, VAE_CKPT, POS_EMB, NEG_EMB) if not os.path.exists(p)]


def load() -> None:
    """Build DiT + VAE on GPU. ~14 GB of state dict; slow."""
    global _runner, _embeds, _load_error
    with _lock:
        if _runner is not None:
            return
        missing = missing_weights()
        if missing:
            _load_error = f"missing checkpoints: {missing} (run ./fetch_weights.sh)"
            raise RuntimeError(_load_error)

        # Upstream configs use `__inherit__:` with repo-relative paths, and
        # `__object__: path: models.dit_v2.nadit` for class resolution, so both cwd
        # and sys.path must be the checkout root.
        os.chdir(SEEDVR_SRC)

        from common.config import load_config  # noqa: PLC0415
        from common.distributed import init_torch  # noqa: PLC0415
        from common.seed import set_seed  # noqa: PLC0415
        from projects.video_diffusion_sr.infer import VideoDiffusionInfer  # noqa: PLC0415

        t0 = time.time()
        init_torch(cudnn_benchmark=False)

        config = load_config(CONFIG)
        config.vae.checkpoint = VAE_CKPT  # upstream hardcodes ./ckpts/ema_vae.pth

        runner = VideoDiffusionInfer(config)
        runner.configure_dit_model(device="cuda", checkpoint=DIT_CKPT)
        runner.configure_vae_model()

        set_seed(int(os.environ.get("SEEDVR2_SEED", "666")), same_across_ranks=True)

        _embeds = {
            "texts_pos": [torch.load(POS_EMB, map_location="cpu")],
            "texts_neg": [torch.load(NEG_EMB, map_location="cpu")],
        }
        _runner = runner
        print(f"[seedvr2] loaded in {time.time() - t0:.1f}s", flush=True)


def is_loaded() -> bool:
    return _runner is not None


def load_error() -> str | None:
    return _load_error


def _build_transform(res_h: int, res_w: int):
    from data.image.transforms.divisible_crop import DivisibleCrop  # noqa: PLC0415
    from data.image.transforms.na_resize import NaResize  # noqa: PLC0415
    from data.video.transforms.rearrange import Rearrange  # noqa: PLC0415
    from torchvision.transforms import Compose, Lambda, Normalize  # noqa: PLC0415

    return Compose(
        [
            NaResize(resolution=(res_h * res_w) ** 0.5, mode="area", downsample_only=False),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisibleCrop((16, 16)),
            Normalize(0.5, 0.5),
            Rearrange("t c h w -> c t h w"),
        ]
    )


@torch.no_grad()
def restore(
    img: Image.Image,
    *,
    scale: float | None = 4.0,
    res_h: int | None = None,
    res_w: int | None = None,
    cfg_scale: float = 1.0,
    cfg_rescale: float = 0.0,
    sample_steps: int = 1,
    seed: int = 666,
) -> tuple[Image.Image, dict]:
    """Restore a single image.

    Give either `scale` (target = input * scale) or an explicit `res_h`/`res_w`.
    """
    if _runner is None:
        raise RuntimeError("model not loaded")
    if sample_steps < 1:
        raise ValueError("sample_steps must be >= 1")

    import numpy as np  # noqa: PLC0415
    from common.distributed import get_device  # noqa: PLC0415
    from common.seed import set_seed  # noqa: PLC0415
    from einops import rearrange  # noqa: PLC0415

    src_w, src_h = img.size
    if res_h is None or res_w is None:
        if not scale or scale <= 0:
            raise ValueError("provide scale > 0, or both res_h and res_w")
        res_h, res_w = int(round(src_h * scale)), int(round(src_w * scale))

    t0 = time.time()
    _runner.config.diffusion.cfg.scale = cfg_scale
    _runner.config.diffusion.cfg.rescale = cfg_rescale
    _runner.config.diffusion.timesteps.sampling.steps = sample_steps
    _runner.configure_diffusion()
    set_seed(seed, same_across_ranks=True)

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    video = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # t=1 c h w
    cond = [_build_transform(res_h, res_w)(video.to(get_device()))]

    # Upstream's GPU/CPU shuffle: only one of DiT/VAE resident at a time.
    _runner.dit.to("cpu")
    _runner.vae.to(get_device())
    cond_latents = _runner.vae_encode(cond)
    _runner.vae.to("cpu")
    _runner.dit.to(get_device())

    embeds = {
        "texts_pos": [e.to(get_device()) for e in _embeds["texts_pos"]],
        "texts_neg": [e.to(get_device()) for e in _embeds["texts_neg"]],
    }

    from .generation import generation_step  # noqa: PLC0415

    samples = generation_step(_runner, embeds, cond_latents=cond_latents)
    _runner.dit.to("cpu")

    sample = samples[0]
    sample = rearrange(sample[:, None], "c t h w -> t c h w") if sample.ndim == 3 else sample
    frame = sample[0]  # single image -> single frame

    out = frame.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
    out = out.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    out_img = Image.fromarray(out)

    torch.cuda.empty_cache()
    meta = {
        "model": os.path.basename(DIT_CKPT),
        "input_size": [src_w, src_h],
        "output_size": [out_img.width, out_img.height],
        "requested_scale": scale,
        "target_res": [res_w, res_h],
        "effective_scale": round(out_img.width / src_w, 4),
        "sample_steps": sample_steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "duration_s": round(time.time() - t0, 2),
    }
    return out_img, meta
