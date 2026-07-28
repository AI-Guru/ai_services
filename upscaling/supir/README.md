# SUPIR — diffusion restoration (the "creative" tier)

[SUPIR](https://github.com/Fanghua-Yu/SUPIR) (Yu et al., [CVPR 2024](https://arxiv.org/abs/2401.13627))
applies SDXL's generative prior to image restoration. It does not interpolate — it **invents**
plausible detail. That makes it the opposite trade-off from the Real-ESRGAN service next door
([`../`](../)), not an upgrade to it, and the choice between them is a product decision rather than
a quality ranking. Measured results below.

Runs as a **separate service on its own port** rather than as a tier inside the Real-ESRGAN
container, because the resource profiles have nothing in common: 15 GB VRAM vs ~0.3 GB, ~11 s vs
0.85 s, 21 GB of checkpoints vs 128 MB.

## Run

```bash
./fetch_weights.sh                    # once — ~21 GB into ./weights (gitignored)
docker compose -f docker-compose.supir-sdxl-fp16-rtx.yml up -d --build
curl -sS http://localhost:11478/health
# {"status":"ok","model":"SUPIR-v0Q",...,"gpu_used_gb":15.04}
```

`/health` returns **503** until the model is genuinely loaded (~35 s) and **503 forever** if loading
failed — so the compose healthcheck cannot mark a broken container healthy. See
[CLAUDE.md](../../CLAUDE.md) on crash-loop detection; a flat-200 health endpoint is the same trap.

## Use

Same route shape as the sibling upscaler, so a client can switch base URL and keep working.

| Field | Default | Meaning |
|---|---|---|
| `upscale` | 2 | output factor |
| `prompt` | `""` | **replaces upstream's LLaVA caption** — describing the subject measurably helps |
| `negative_prompt` | (SUPIR's default) | what to suppress |
| `edm_steps` | 50 | diffusion steps; the main quality/time dial |
| `s_cfg` | 4.0 | guidance — lower = more faithful, higher = more invention |
| `s_stage2` | 1.0 | control strength; drop below 1.0 for heavier degradation |
| `seed` | 1234 | reproducibility |
| `color_fix_type` | `Wavelet` | `Wavelet` / `AdaIn` / `None` |

```bash
# multipart -> PNG bytes
curl -sS -X POST http://localhost:11478/upscale \
  -F "file=@input.png" -F "upscale=4" -F "prompt=a gothic cathedral interior" -o out.png

# JSON b64 in/out — same envelope as the generation services and ../app
curl -sS -X POST http://localhost:11478/upscale/json -H "Content-Type: application/json" \
  -d "{\"image\":\"$(base64 -w0 input.png)\",\"upscale\":4,\"edm_steps\":50}" \
  | jq -r '.data[0].b64_json' | base64 -d > out.png
```

Calls are **synchronous and take tens of seconds** — set generous client timeouts. An async
job API is the honest long-term shape here; it wasn't built because the one-step diffusion
line (OSEDiff / TSD-SR / FiDeSR) may make it unnecessary before it becomes painful.

LLaVA (13B auto-captioner) is deliberately not loaded — `prompt` covers the same ground for
~14 GB less weight.

## Measured on this box (RTX PRO 6000, 96 GB)

Method: take a high-quality source, degrade it ÷4 + JPEG q40, restore ×4, score against the
**undegraded original**. Side-by-side 1:1 crops:
[church](../comparison_out/montage_church1.png) ·
[photo](../comparison_out/montage_testimage.png) ·
[diagram](../comparison_out/montage_agentworld-roles.png).

```bash
python3 ../compare_upscalers.py --images ../../church1.png --scale 4 --edm-steps 50
```

Regenerates the montages plus a full base64-embedded `comparison.html` and the full-size outputs
(both gitignored — 18 MB, against a repo `comparison*.html` convention of 14–36 KB). Seeded, so the
numbers below reproduce exactly.

| Image | Method | PSNR dB | SSIM | sec |
|---|---|---:|---:|---:|
| church1 (AI-generated) | lanczos | 23.91 | 0.7138 | 0.01 |
| | realesrgan | **24.27** | **0.7237** | 0.31 |
| | supir | 21.14 | 0.6095 | 11.12 |
| testimage (photo) | lanczos | **22.53** | **0.6505** | 0.00 |
| | realesrgan | 21.64 | 0.6450 | 0.18 |
| | supir | 19.80 | 0.5464 | 10.95 |
| agentworld-roles (diagram) | lanczos | **25.22** | 0.9102 | 0.01 |
| | realesrgan | 25.09 | **0.9566** | 0.28 |
| | supir | 24.46 | 0.9147 | 22.22 |

**SUPIR loses every single distortion measurement — and that is the expected result.** PSNR and
SSIM reward pixel-fidelity; Blau & Michaeli ([CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html))
proved distortion and perceptual quality are formally at odds. Look at the crops in the HTML report
and SUPIR is plainly the sharpest of the three on photographic content. The numbers and the eyes
disagree *by construction*, which is the whole point of running both.

Two findings that are not just the trade-off, though:

**1. Real-ESRGAN lost to plain Lanczos on the photo** (21.64 vs 22.53 dB). A 5-year-old GAN is not
unconditionally better than classical resampling on a fidelity metric — worth remembering before
reaching for a model at all.

**2. SUPIR corrupts text.** On the diagram, the word **“action” came back as “nstion”.** It is not
blurry-but-right; it is crisp and *wrong*, which is worse, because nothing downstream can tell. This
is precisely the failure mode [Hallucination Score (arXiv 2507.14367)](https://arxiv.org/abs/2507.14367)
argues existing metrics can't see — note SUPIR's SSIM on that image (0.9147) looks perfectly
respectable while the semantic content is destroyed.

### Choosing

- **Documents, diagrams, screenshots, anything with text, anything evidential** → Real-ESRGAN, or
  Lanczos. Never SUPIR.
- **Photos and heavily degraded material where a plausible reconstruction beats a faithful blur**
  → SUPIR, with a `prompt`.
- **Batch / latency-sensitive / chained after generation** → Real-ESRGAN; it is ~35–70× faster.

## Getting it to run on Blackwell

SUPIR's `requirements.txt` is pinned to early-2023 versions and the repo has been dormant since
2025-05-12. Three things had to be resolved; all are handled in the image, listed here because they
will recur for anyone building SDXL-era code on SM_120.

1. **`triton==2.1.0` / `xformers`** — both drag torch off the cu128 wheel Blackwell needs. Dropped:
   triton isn't imported directly, and `sgm/modules/attention.py` has a working non-xformers
   fallback. The Dockerfile **asserts torch is still 2.7.x+cu128 after pip runs**, so a transitive
   downgrade fails the build instead of surfacing later as `no kernel image is available`.

2. **Tiled VAE hard-crashed without xformers** — two stacked upstream bugs, patched by
   [`patch_tilevae.py`](patch_tilevae.py):
   - `tilevae.py:364` reads `if is_xformers_available:` — testing the imported **function object**,
     which is always truthy. So it always dispatched to xformers → `NameError: name 'xformers' is
     not defined`.
   - Adding the missing `()` is *wrong*: the `elif` branch it then reaches
     (`attn_forward_new_pt2_0`) expects a diffusers `Attention` module, but sgm passes a
     `MemoryEfficientAttnBlock` → `'MemoryEfficientAttnBlock' object has no attribute 'group_norm'`.

     Fixed instead by swapping the single op inside the function that already has the right tensor
     plumbing: `q/k/v` are `(B, HW, C)` single-head, so `F.scaled_dot_product_attention` is a true
     drop-in for `xformers.ops.memory_efficient_attention`.

3. **`open-clip-torch` is pinned to 2.24.0** — the only pin kept. 3.x rewrote
   `create_model_and_transforms` and the text-transformer mask handling; sgm drives the transformer
   directly and dies with `RuntimeError: The shape of the 2D attn_mask is torch.Size([77, 77]), but
   should be (1, 1)`. `transformers` 5.x, by contrast, needed no pin at all despite upstream asking
   for 4.28.1.

## Checkpoints

`./weights` is gitignored. `fetch_weights.sh` pulls:

| File | Size | Source |
|---|---|---|
| `SUPIR-v0Q_fp16.safetensors` | 2.7 GB | `Kijai/SUPIR_pruned` — **community mirror** |
| `sd_xl_base_1.0_0.9vae.safetensors` | 6.9 GB | `stabilityai` (first-party) |
| `CLIP-ViT-bigG-14.../open_clip_pytorch_model.bin` | 10.2 GB | `laion` (first-party) |
| `clip-vit-large-patch14/` | 1.7 GB | `openai` (first-party) |

The SUPIR weights are the one **unvetted community quant** here, flagged per the repo convention.
Upstream distributes the originals only via Google Drive and Baidu Netdisk, neither scriptable;
Kijai's pruned fp16 is the de-facto mirror the ComfyUI ecosystem uses. `SUPIR-v0F` (light
degradation) is not fetched by default — set `SUPIR_SIGN=F` and add it if you want that variant.
