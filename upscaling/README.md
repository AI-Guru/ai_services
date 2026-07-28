# Image upscaler — Real-ESRGAN x4 (thin FastAPI service)

Companion to [`models/z-image`](../models/z-image/) (generation). Upscaling is **not** part of the
OpenAI image API, and **vLLM-Omni has no super-resolution models in its registry** — so this is a
deliberately small dedicated service rather than something bolted onto the generation endpoint.

POST an image → get a 4× PNG back. Built on [spandrel](https://github.com/chaiNNer-org/spandrel)
(the model loader ComfyUI uses), which auto-detects the architecture from a bare `.pth` and avoids the
legacy `realesrgan`/`basicsr` dependency rot that breaks on modern torch / Blackwell.

## Run

```bash
cd upscaling
docker compose --env-file ../.env -f docker-compose.upscaler-realesrgan-rtx.yml up -d --build
docker logs -f upscaler   # first boot downloads RealESRGAN_x4plus.pth into ./weights
```

## Use

`GET /health` → readiness + available native scales (`loaded` is what's actually resident in VRAM,
which trails `native_scales` until a lazily-loaded model is first used):

```bash
curl -sS http://localhost:11477/health
# {"status":"ok","native_scales":[2,4],"default_scale":4,"loaded":[4]}
```

`POST /upscale` → multipart `file`, returns a PNG. Two optional form fields parametrize the factor:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `scale` | `2` or `4` | `4` | Native Real-ESRGAN model to run. |
| `outscale` | float > 0 | — | Arbitrary factor (e.g. `1.5`, `3`, `8`). Overrides `scale`: runs the nearest native model, then Lanczos-resamples to the exact factor. |

The response carries `X-Native-Scale`, `X-Effective-Scale`, and `X-Output-Size` headers.

```bash
# default x4
curl -sS -X POST http://localhost:11477/upscale -F "file=@input.png" -o out_x4.png

# native x2
curl -sS -X POST http://localhost:11477/upscale -F "file=@input.png" -F "scale=2" -o out_x2.png

# arbitrary x3 (native x4 -> resample)
curl -sS -X POST http://localhost:11477/upscale -F "file=@input.png" -F "outscale=3" -o out_x3.png
```

From Python:

```python
import requests
with open("input.png", "rb") as f:
    r = requests.post(
        "http://localhost:11477/upscale",
        files={"file": f},
        data={"scale": 2},          # or data={"outscale": 3}
    )
open("output.png", "wb").write(r.content)
```

The native ×2 model loads lazily on first use (×4 is preloaded at boot; set `UPSCALER_PRELOAD=2,4`
in the compose to preload both).

`POST /upscale/json` → JSON-in/JSON-out twin, for chaining with the generation services without a
round-trip through disk. Same `scale` / `outscale` semantics; body is `{"image": "<base64>", ...}`
(a `data:image/png;base64,` prefix is tolerated) and the reply reuses the generation services'
OpenAI-shaped envelope, with the scale metadata alongside instead of in headers:

```jsonc
{
  "created": 1753689600,
  "data": [{"b64_json": "..."}],
  "native_scale": 4, "effective_scale": 3.0,
  "input_size": [512, 512], "output_size": [1536, 1536]
}
```

Multipart `/upscale` stays the primary route — it avoids the ~33% base64 tax on both legs, so prefer
it whenever you already hold the bytes. Use `/upscale/json` when the image is already base64 because
it came out of a generation call.

### Chaining with generation

Generate small + fast on [`z-image`](../models/z-image/), then upscale — cheaper than generating large.
Both services speak `b64_json`, so it's a two-call JSON pipeline:

```bash
curl -sS -X POST http://localhost:11476/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"z-image-turbo","prompt":"a tiny vintage camera on a desk","size":"512x512","num_inference_steps":8,"guidance_scale":1.0}' \
  | jq '{image: .data[0].b64_json, scale: 4}' \
  | curl -sS -X POST http://localhost:11477/upscale/json -H "Content-Type: application/json" -d @- \
  | jq -r '.data[0].b64_json' | base64 -d > big.png   # 512 -> 2048
```

## Swapping the model

Real-ESRGAN x2/x4 are fast, faithful, and great for batch work — but plasticky on heavily AI-generated
images. To add or swap a model, edit `NATIVE_WEIGHTS` in [`app/upscaler.py`](app/upscaler.py): drop any
**ESRGAN / Real-ESRGAN / SwinIR / DAT-family `.pth`** keyed by its native scale (spandrel auto-detects the
architecture). Weights download to `./weights` on first use.

For **diffusion restoration** — which *invents* detail rather than interpolating it — see the sibling
service in [`supir/`](supir/). It runs SUPIR (CVPR 2024) on its own port with its own 21 GB of
checkpoints, rather than as a tier inside this container: 15 GB VRAM vs ~0.3 GB and ~11 s vs 0.85 s
have no business sharing a process. This service intentionally stays a one-call upscaler.

(That note previously said "use ComfyUI instead". Still true for SeedVR2 and for anything wanting a
node graph — but SUPIR turned out to be servable as a plain one-call endpoint, so it is.)

## Which one to use

Measured ×4 restoration from ÷4 + JPEG q40 degradation, scored against the undegraded original by
[`compare_upscalers.py`](compare_upscalers.py). Side-by-side 1:1 crops:
[church](comparison_out/montage_church1.png) ·
[photo](comparison_out/montage_testimage.png) ·
[diagram](comparison_out/montage_agentworld-roles.png):

| | PSNR / SSIM | speed | fails at |
|---|---|---|---|
| **Lanczos** (built in, no model) | 22.5–25.2 dB | instant | detail — it can only blur |
| **Real-ESRGAN** (this service) | 21.6–25.1 dB, best SSIM on graphics | ~0.3 s | plasticky on AI-generated input |
| **SUPIR** ([`supir/`](supir/)) | **lowest on every image** | ~11–22 s | **corrupts text** — see below |

SUPIR losing every distortion metric is the expected result, not a defect: PSNR/SSIM reward
pixel-fidelity and generative restoration trades exactly that away for perceptual quality (Blau &
Michaeli, CVPR 2018). Visually it is the sharpest of the three on photos.

Two results worth knowing before choosing:

- On the photo, **Real-ESRGAN scored below plain Lanczos** (21.64 vs 22.53 dB). A model is not
  automatically better than resampling.
- On a diagram, **SUPIR rendered the word "action" as "nstion"** — crisply, and wrong. Do not point
  it at documents, screenshots, or anything evidential. Its SSIM on that image (0.9147) looks fine,
  which is the problem.

## Status / measured on this box (RTX PRO 6000, 96 GB)

✅ **Verified working** — 384×384 → 1536×1536 (×4) in **~0.85 s**, HTTP 200, valid PNG.

- Base image matches `open-genmoji` (CUDA 12.8 + torch 2.7 cu128) — the SM_120 recipe already proven on
  this box. Verify the container reaches `healthy` before trusting it (see CLAUDE.md crash-loop waiter).
- Whole-image fp32 inference; no tiling (96 GB makes it unnecessary short of gigapixel inputs).
- VRAM footprint is negligible (~hundreds of MB); coexists with the generation service.
- `./weights/` is gitignored — checkpoints are downloaded, not committed.
