# image-generation — self-hosted text-to-image

Open-weights text-to-image models served behind an **OpenAI-compatible image API**
(`POST /v1/images/generations`), with a Gradio frontend. Same spirit as `models/`:
each model is a `docker-compose.<engine>-<variant>.yml`, one model resident on the
single GPU at a time.

Unlike the LLM families (vLLM ships its own server), 🤗 **diffusers is a library, not
a server** — so a small shared FastAPI wrapper turns any diffusers text-to-image
pipeline into an OpenAI-style endpoint. One image serves every model; the concrete
pipeline is selected by env vars.

## Layout

```
image-generation/
├── diffusers-server/   reusable OpenAI-compatible server (server.py + Dockerfile)
├── gradio-app/         frontend: model dropdown, batch size, gallery
├── z-image/            Z-Image-Turbo  (6B,  Apache-2.0)
└── qwen-image/         Qwen-Image     (20B, Apache-2.0)
```

## Models

| Model | Params / Arch | License | VRAM (n=1) | Speed @1024² | Defaults | Port |
|-------|---------------|---------|-----------|--------------|----------|------|
| **Z-Image-Turbo** | 6B single-stream DiT (S3-DiT), 8-step distilled | Apache-2.0 | ~23 GB | ~2.8 s (9 steps) | steps 9, CFG 0 | 8101 |
| **Qwen-Image** | 20B MMDiT | Apache-2.0 | ~56 GB | ~28 s (50 steps) | steps 50, true-CFG 4 | 8102 |

Both are first-party Apache-2.0 checkpoints (no token, no commercial restrictions).
Z-Image-Turbo ranked #1 open-weights on the Artificial Analysis Image Arena (mid-2026);
Qwen-Image leads open-weights on complex / Chinese **text rendering**. Verified on the
RTX PRO 6000 Blackwell (sm_120, torch 2.11+cu130).

## Run

One model at a time (single shared 96 GB card). Bring a model up first (it creates the
`ai-image-network`), then the frontend:

```bash
# Z-Image (fast default)
docker compose -f z-image/docker-compose.diffusers-6b-rtx.yml up -d --build

# …or Qwen-Image (stop the other model first)
docker compose -f qwen-image/docker-compose.diffusers-20b-rtx.yml up -d --build

# frontend → http://localhost:7860
docker compose -f gradio-app/docker-compose.yml up -d --build
```

To swap models: `docker stop z-image-turbo && docker start qwen-image` (or vice-versa).

## API

```bash
curl -s -X POST http://localhost:8101/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"z-image-turbo","prompt":"a red panda barista","size":"1024x1024","n":1,"seed":7}' \
  | python3 -c "import sys,json,base64;d=json.load(sys.stdin);open('o.png','wb').write(base64.b64decode(d['data'][0]['b64_json']))"
```

Request fields: `prompt`, `n` (batch — generated in one parallel pass), `size` (`WxH`),
`num_inference_steps`, `guidance_scale`, `negative_prompt`, `seed`. Omitted fields fall
back to the model's defaults. Response: `{"data":[{"b64_json": ...}]}` (PNG, base64).
Also: `GET /health`, `GET /v1/models`, `GET /v1/config` (per-model defaults — the Gradio
sliders auto-adapt from this when you switch models).

## Shared server — env config

`diffusers-server/` is model-agnostic; each compose sets:

| Env | Purpose |
|-----|---------|
| `MODEL_ID` / `PIPELINE_CLASS` | HF repo + diffusers class (e.g. `QwenImagePipeline`) |
| `SERVED_MODEL_NAME` | name in `/v1/models` and matched on requests |
| `DEFAULT_STEPS` / `DEFAULT_GUIDANCE` | generation defaults |
| `GUIDANCE_PARAM` | `guidance_scale` (default) or `true_cfg_scale` (Qwen-Image) |
| `DEFAULT_NEG_PROMPT` | negative prompt default (Qwen needs `" "` min) |
| `DEVICE_MAP` | e.g. `cuda` — stream weights straight to GPU (low host-RAM hosts) |
| `LOW_CPU_MEM_USAGE` | `from_pretrained` flag |

> **Note** — the host has only ~30 GB RAM, so the 20B Qwen-Image is loaded with
> `DEVICE_MAP=cuda` to stream weights directly to the 96 GB GPU instead of
> materializing the full model in CPU RAM (which OOMs). Z-Image (6B) loads normally.

## Notes / conventions

- Base image: `vllm/vllm-openai:cu130-nightly` — it already ships a Blackwell/sm_120
  PyTorch, so the CUDA stack is proven. diffusers is installed from source (`--no-deps`)
  because newer pipelines (ZImagePipeline) ship ahead of the last PyPI release.
- Batch size scales VRAM (~+6.8 GB per extra image on Qwen@1024); the UI surfaces an
  OOM as a clean error.
- `*/out/` (smoke/API/batch test images) is gitignored.
