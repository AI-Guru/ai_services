# Qwen3.5-35B-A3B on NVIDIA DGX Spark

Runs [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) via vLLM on the DGX Spark (GB10/Blackwell, 128 GB unified memory), exposing an OpenAI-compatible API.

Based on: https://github.com/adadrag/qwen3.5-dgx-spark

## Requirements

- NVIDIA DGX Spark (GB10, 128 GB unified memory)
- Docker + NVIDIA Container Toolkit
- `vllm/vllm-openai:cu130-nightly` — the standard NVIDIA container ships vLLM 0.13.0 which does not support Qwen3.5; the nightly build (v0.16.0+) is required

## Quick start

```bash
# Default — BF16, 262K context, multimodal (port 8000)
docker compose up -d

# Watch startup (model load takes 3–5 min on first run)
docker compose logs -f
```

API is ready when logs show `Application startup complete`.

## Service variants

All variants serve on **port 8000**. Only run one at a time.

| Variant | Command | Model | Weights | Notes |
|---|---|---|---|---|
| Default | `docker compose up -d` | `Qwen3.5-35B-A3B` | ~70 GB BF16 | Recommended starting point |
| Extended context | `docker compose -f docker-compose.1m.yaml up -d` | `Qwen3.5-35B-A3B` | ~70 GB BF16 | ~1M tokens via YARN RoPE override |
| Text-only | `docker compose -f docker-compose.text-only.yaml up -d` | `Qwen3.5-35B-A3B` | ~70 GB BF16 | Vision encoder disabled |
| FP8 | `docker compose -f docker-compose.fp8.yaml up -d` | `Qwen3.5-35B-A3B-FP8` | ~35 GB FP8 | 3× more KV cache; W8A8 on GB10 |

All variants share the same named Docker volume (`qwen35_huggingface_cache`) so weights are only downloaded once per model variant.

### FP8 notes

The `fp8` profile uses the official Qwen pre-quantized model ([Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) rather than on-the-fly quantization. GB10 (Blackwell, CC 12.1) runs block-wise FP8 as W8A8 natively with no throughput penalty. Halving the weight footprint from ~70 GB to ~35 GB leaves ~87 GB available for KV cache at 0.90 utilisation — roughly 3× more than the BF16 default.

`--kv-cache-dtype fp8_e4m3` is intentionally omitted: Qwen3.5's hybrid MoE+Mamba architecture triggers a vLLM bug at runtime ([vllm-project/vllm#26646](https://github.com/vllm-project/vllm/issues/26646)). Weight-only FP8 already provides the main memory saving.

### MXFP4

No vLLM-compatible MXFP4 checkpoint exists for this model yet. A GGUF variant is available ([noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF](https://huggingface.co/noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF), ~22 GB) but requires llama.cpp. Additionally, vLLM's native NVFP4 kernel support on GB10 (SM12.x) is still maturing ([vllm-project/vllm#31085](https://github.com/vllm-project/vllm/issues/31085)).

## Memory layout (DGX Spark, 128 GB unified)

| Variant | Weights | `gpu_memory_utilization` | KV cache headroom |
|---|---|---|---|
| BF16 (default) | ~70 GB | 0.80 | ~28 GB |
| BF16 (text-only) | ~67 GB | 0.85 | ~31 GB |
| BF16 (1M context) | ~70 GB | 0.95 | ~52 GB |
| FP8 | ~35 GB | 0.90 | ~87 GB |

On DGX Spark, CPU and GPU share the same physical memory pool. High RAM utilisation on the dashboard during model load is expected — the weights live in unified memory.

## API

All profiles expose an OpenAI-compatible API. The model is served as `qwen3.5-35b`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Other useful endpoints:

```bash
curl http://localhost:8000/health          # liveness
curl http://localhost:8000/v1/models       # confirm model name
```

## Test scripts

```bash
pip install openai

# Chat latency — TTFT and tokens/sec
python test_chat.py
python test_chat.py --runs 5
python test_chat.py --prompt "Your prompt here" --base-url http://localhost:8000/v1

# Tool calling — single, parallel, and chained scenarios
python test_tools.py
python test_tools.py --scenario single
python test_tools.py --base-url http://localhost:8000/v1
```

## Environment variables

Create a `.env` file alongside `docker-compose.yaml` to override defaults:

```dotenv
HF_TOKEN=hf_...              # required for gated models; not needed for Qwen3.5
HF_HUB_OFFLINE=1             # set after first successful download to block runtime fetches
VLLM_LOGGING_LEVEL=WARNING   # reduce log noise in production
```

## Troubleshooting

**Container stays `unhealthy`**
The 70 GB model takes longer than the 300 s healthcheck `start_period` on first run while the weights are being downloaded. Check actual progress with `docker compose logs -f` and wait for `Application startup complete`. The container will self-heal once the API is up.

**High swap usage**
Normal during model load on DGX Spark. Swap should stabilise once weights are fully resident in unified memory. If swap keeps climbing, lower `--gpu-memory-utilization` (e.g. from `0.80` to `0.75`).

**`fp8e4nv not supported` error**
This is the KV cache FP8 bug ([#26646](https://github.com/vllm-project/vllm/issues/26646)). The FP8 profile already omits `--kv-cache-dtype`; if you have added it manually, remove it.

**PyTorch CUDA capability warning (GB10 / CC 12.1)**
`Found GPU0 NVIDIA GB10 which is of cuda capability 12.1. Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)` — this is a warning only, not an error. The model runs correctly.
