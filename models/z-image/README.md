# Z-Image-Turbo — self-hosted text-to-image (OpenAI-compatible)

[Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) served via
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) behind the OpenAI DALL-E image API.

- **Model**: 6B Single-Stream DiT (S3-DiT), 8-step distilled. First-party (Alibaba), **Apache 2.0** → commercial-safe.
- **Why this one**: ~95% of FLUX-class quality at a fraction of the compute; ranked #1 open model on
  Artificial Analysis Image Arena (reportedly above FLUX.2[dev], HunyuanImage 3.0, Qwen-Image).
  Trivial footprint on the 96 GB card (~16 GB fp16), so it's the workhorse default. Use **FLUX.2[dev]
  via ComfyUI** when you need 4-MP / max fidelity.
- **Not an upscaler**: vLLM-Omni has no super-resolution models in its registry. For upscaling use the
  sibling [`upscaling/`](../../upscaling/) service.

## Run

```bash
cd models/z-image
docker compose --env-file ../../.env -f docker-compose.vllm-omni-turbo-rtx.yml up -d
# watch it actually come up (don't trust Running==true — see CLAUDE.md crash-loop note):
docker logs -f z-image-turbo
```

API: `http://localhost:11476/v1` · served model name: `z-image-turbo`

## Generate

Quick smoke test (decodes the b64 response to a PNG for you):

```bash
./test_image.sh "a dragon over the Green Mountains of Vermont" out.png
```

### `POST /v1/images/generations`

```bash
curl -sS -X POST http://localhost:11476/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-image-turbo",
    "prompt": "a photorealistic red panda barista pulling an espresso shot",
    "size": "1024x1024",
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "negative_prompt": "blurry, low quality, distorted, extra fingers",
    "seed": 42,
    "n": 1
  }' \
  | python3 -c "import json,sys,base64; d=json.load(sys.stdin); open('out.png','wb').write(base64.b64decode(d['data'][0]['b64_json']))"
```

**Response shape** (OpenAI-compatible): `data[0]` carries `b64_json`, `url`, and `revised_prompt`.
Decode `b64_json` to get the PNG (as above), or pass `"response_format": "url"` to fetch via `url`.

Beyond the OpenAI fields (`prompt`, `size`, `n`, `response_format`), vLLM-Omni accepts
`negative_prompt`, `num_inference_steps`, `guidance_scale`, `seed`, `flow_shift`, and per-request LoRA.
Image-to-image / editing is at `POST /v1/images/edits` (multipart, send an `image` + `prompt`).

### From Python (OpenAI SDK)

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:11476/v1", api_key="none")
r = client.images.generate(
    model="z-image-turbo",
    prompt="a serene mountain lake at dawn",
    size="1024x1024",
    extra_body={"num_inference_steps": 8, "guidance_scale": 1.0, "seed": 7},
)
open("out.png", "wb").write(base64.b64decode(r.data[0].b64_json))
```

> **Distilled-model settings**: keep `num_inference_steps` ≈ 8 and `guidance_scale` ≈ 1.0 (CFG off).
> Treating it like a base model (40 steps, CFG 7.5) makes output *worse*.

## Gradio test app

A small 3-tab playground (Generate / Edit / Upscale) that drives the generation and upscaler endpoints
over HTTP — [`gradio_app.py`](gradio_app.py). Pure client, no GPU.

```bash
pip install -e ".[gradio]"            # from repo root (deps live in the root pyproject)
python models/z-image/gradio_app.py   # -> http://localhost:7861
```

Point it at a remote host with env vars:

```bash
GEN_URL=http://192.168.0.161:11476 UPSCALE_URL=http://192.168.0.161:11477 \
  python models/z-image/gradio_app.py
```

The **Edit** tab uses `POST /v1/images/edits` (Z-Image img2img — verified working); the **Upscale**
tab calls the [`upscaling/`](../../upscaling/) service.

## LibreChat

Z-Image speaks the OpenAI image API, so it wires into LibreChat's image-generation tooling rather than
the `endpoints.custom` (chat) block in [`librechat/librechat.yaml`](../../librechat/librechat.yaml).
Pointed at `http://host.docker.internal:11476/v1`. Wiring is a follow-up — confirm LibreChat's current
image-gen config mechanism before editing.

## Status / measured on this box (RTX PRO 6000, 96 GB)

✅ **Verified working** — `vllm/vllm-omni:v0.22.0` loads cleanly on SM_120 and serves the OpenAI image API.

| Metric | Value |
|---|---|
| Cold first generation (1024², 8 steps) | ~3.2 s |
| Warm generation (1024², 8 steps) | **~2.97 s**, very stable |
| VRAM resident | ~12 GB (shares the card comfortably; ~28 GB with the upscaler also up) |
| Startup to `healthy` | ~220 s (image pull + 6B weight download, first run only) |

Caveats:
- This image has **no ENTRYPOINT** — the compose spells out `vllm serve … --omni` in full. Don't "simplify"
  the command to a bare model arg; it'll try to exec the model id as a binary.
- NVFP4 diffusion on SM_120 is still an open RFC
  ([vllm-omni#1959](https://github.com/vllm-project/vllm-omni/issues/1959)); this config runs the default
  fp16/bf16 path, which is plenty for a 6B model.
