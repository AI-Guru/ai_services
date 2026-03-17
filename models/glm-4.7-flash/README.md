# GLM-4.7-Flash

Local inference for GLM-4.7-Flash with multiple backends and quantization options.

## Overview

- **Model**: GLM-4.7-Flash (30B MoE, ~3.6B active parameters)
- **Context Window**: Up to 202,752 tokens
- **Port**: 11346
- **API**: OpenAI-compatible (`/v1/chat/completions`)

## Quick Start

Pick a backend and quantization, then run the corresponding compose file:

```bash
# Create the shared cache volume (one-time)
docker volume create glm47_huggingface_cache

# llama.cpp Q4 (recommended — low VRAM, simple)
docker compose -f docker-compose.llama-q4-rtx.yml up -d

# llama.cpp Q8 (higher quality, more VRAM)
docker compose -f docker-compose.llama-q8-rtx.yml up -d

# vLLM FP8 (high throughput, continuous batching)
docker compose -f docker-compose.vllm-fp8.yml up -d

# vLLM BF16 (full precision, most VRAM)
docker compose -f docker-compose.vllm-bf16.yml up -d
```

Models are downloaded automatically on first start.

## Configurations

### llama.cpp

| File | Quant | Base VRAM | With 202K Context |
|------|-------|-----------|--------------------|
| `docker-compose.llama-q4-rtx.yml` | Q4_K_XL | ~18 GB | ~38 GB |
| `docker-compose.llama-q8-rtx.yml` | Q8_K_XL | ~33 GB | ~58 GB |

Uses `ghcr.io/ggml-org/llama.cpp:server-cuda` with automatic GGUF download from
[unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF).

Server parameters: `--temp 0.7 --top-p 1.0 --repeat-penalty 1.0` (tool-calling optimized).
For general use, adjust to `--temp 1.0 --top-p 0.95`.

### vLLM

| File | Precision | VRAM | Context |
|------|-----------|------|---------|
| `docker-compose.vllm-fp8.yml` | FP8-Dynamic | ~30 GB | 200K |
| `docker-compose.vllm-bf16.yml` | BF16 | ~70 GB | 131K |

Uses a custom image (`vllm-glm47-cu130`) built from inline Dockerfile with vLLM nightly cu130
for GLM-4.7 support. The image is built automatically on first `docker compose up`.

Both variants include tool-calling (`--tool-call-parser glm47`) and reasoning
(`--reasoning-parser glm45`) support.

### Backend Comparison

| Aspect | llama.cpp | vLLM |
|--------|-----------|------|
| Quantization | Q4_K_XL / Q8_K_XL | FP8-Dynamic / BF16 |
| Parallelization | Fixed slots | Continuous batching |
| Throughput | ~40–80 tok/s | ~250+ tok/s |
| Concurrent requests | Limited | Scales with memory |
| Setup | Pull-and-run | First build ~10 min |

## Usage

### Test the server

```bash
curl http://localhost:11346/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7-flash",
    "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}]
  }'
```

### Python client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11346/v1",
    api_key="sk-no-key-required"
)

response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## References

- [GLM-4.7-Flash-GGUF on HuggingFace](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [vLLM](https://github.com/vllm-project/vllm)
