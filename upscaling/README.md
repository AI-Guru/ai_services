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

`GET /health` → readiness + native scale factor:

```bash
curl -sS http://localhost:11477/health
# {"status":"ok","model":"RealESRGAN_x4plus.pth","scale":4}
```

`GET /health` reports available native scales:

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

### Chaining with generation

Generate small + fast on [`z-image`](../models/z-image/), then upscale — cheaper than generating large:

```bash
curl -sS -X POST http://localhost:11476/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"z-image-turbo","prompt":"a tiny vintage camera on a desk","size":"512x512","num_inference_steps":8,"guidance_scale":1.0}' \
  | python3 -c "import json,sys,base64; open('small.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
curl -sS -X POST http://localhost:11477/upscale -F "file=@small.png" -o big.png   # 512 -> 2048
```

## Swapping the model

Real-ESRGAN x2/x4 are fast, faithful, and great for batch work — but plasticky on heavily AI-generated
images. To add or swap a model, edit `NATIVE_WEIGHTS` in [`app/upscaler.py`](app/upscaler.py): drop any
**ESRGAN / Real-ESRGAN / SwinIR / DAT-family `.pth`** keyed by its native scale (spandrel auto-detects the
architecture). Weights download to `./weights` on first use.

For **diffusion restoration** (SUPIR, SeedVR2) — which *invents* detail rather than interpolating it —
use ComfyUI instead. It's heavier, stateful, and a different API shape; this service intentionally stays
a one-call upscaler. Decision rationale lives in the repo discussion, not here.

## Status / measured on this box (RTX PRO 6000, 96 GB)

✅ **Verified working** — 384×384 → 1536×1536 (×4) in **~0.85 s**, HTTP 200, valid PNG.

- Base image matches `open-genmoji` (CUDA 12.8 + torch 2.7 cu128) — the SM_120 recipe already proven on
  this box. Verify the container reaches `healthy` before trusting it (see CLAUDE.md crash-loop waiter).
- Whole-image fp32 inference; no tiling (96 GB makes it unnecessary short of gigapixel inputs).
- VRAM footprint is negligible (~hundreds of MB); coexists with the generation service.
- `./weights/` is gitignored — checkpoints are downloaded, not committed.
