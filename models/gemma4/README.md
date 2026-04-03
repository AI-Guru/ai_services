# Gemma 4 Self-Hosted Inference

Runs Google Gemma 4 models via llama.cpp or vLLM on NVIDIA GPUs, exposing an OpenAI-compatible API.

Released April 2, 2026. First Gemma family under Apache 2.0 license. Built from Gemini 3 research.

## Models

| Model | Architecture | Total | Active | Context | Modalities |
|-------|-------------|-------|--------|---------|------------|
| **E2B** | Dense | 5.1B | 2.3B | 128K | Text, Image, Audio |
| **E4B** | Dense | 8B | 4.5B | 128K | Text, Image, Audio |
| **26B-A4B** | MoE (128 experts, top-8+1) | 25.2B | 3.8B | 256K | Text, Image, Video |
| **31B** | Dense | 30.7B | 30.7B | 256K | Text, Image, Video |

All configs serve text-only (`--no-mmproj` for llama.cpp). Multimodal projectors are available in the GGUF repos if needed.

## Requirements

- NVIDIA GPU with 24+ GB VRAM (RTX PRO 6000 96 GB recommended)
- Docker + NVIDIA Container Toolkit
- `ghcr.io/ggml-org/llama.cpp:server-cuda` (llama.cpp) or `vllm/vllm-openai:gemma4-cu130` (vLLM)
- HuggingFace token (Gemma 4 is gated — set `HF_TOKEN` in `.env`)

## Port Assignments

| Port | Model |
|------|-------|
| 11450 | Gemma 4 E2B |
| 11451 | Gemma 4 E4B |
| 11452 | Gemma 4 26B-A4B |
| 11453 | Gemma 4 31B |

## Which variant should I use?

```
What do you need?
├─ Fastest throughput (MoE, ~4B active)?
│  └─ llama.cpp ────────────── llama-26b-rtx.yml (196 tok/s, 131K ctx)
│
├─ Small/edge model?
│  ├─ E2B (fastest) ────────── llama-e2b-rtx.yml (276 tok/s, 128K ctx)
│  └─ E4B (smarter) ────────── llama-e4b-rtx.yml (171 tok/s, 128K ctx)
│
├─ Strongest dense model?
│  ├─ llama.cpp (throughput) ── llama-31b-rtx.yml (65 tok/s, 262K ctx)
│  └─ vLLM NVFP4 (latency) ── vllm-31b-nvfp4-rtx.yml (39 tok/s, 95ms TTFT)
│
└─ Need tool calling / structured output?
   └─ vLLM NVFP4 ──────────── vllm-31b-nvfp4-rtx.yml (Blackwell required)
```

## Performance — RTX PRO 6000 Blackwell (96 GB GDDR7)

| Compose file | Backend | Model | Quant | tok/s | TTFT |
|---|---|---|---|---|---|
| `llama-e2b-rtx.yml` | llama.cpp | E2B dense (2.3B eff.) | Q8_0 | **276.3** | ~2.5 s |
| `llama-26b-rtx.yml` | llama.cpp | 26B MoE (3.8B active) | UD-Q4_K_XL | **196.2** | ~3.8 s |
| `llama-e4b-rtx.yml` | llama.cpp | E4B dense (4.5B eff.) | Q8_0 | **171.0** | ~1.8 s |
| `llama-31b-rtx.yml` | llama.cpp | 31B dense (30.7B) | Q4_K_M | 64.6 | ~9.0 s |
| `vllm-31b-nvfp4-rtx.yml` | vLLM NVFP4 | 31B dense (30.7B) | NVFP4 | 39.3 | **95 ms** |

Benchmarked with `test_chat.py --runs 3 --warmup --no-think` on default prompt.

### Key observations

- **26B MoE is the sweet spot** — 196 tok/s rivals Qwen3.5-35B (194 tok/s) while achieving ~97% of 31B quality on benchmarks.
- **vLLM NVFP4** trades throughput for latency: 39 tok/s but 95ms TTFT vs 9s on llama.cpp. Better for interactive / agentic use.
- **E2B at 276 tok/s** is the fastest Gemma 4 variant — useful for lightweight tasks where quality is secondary.

## Service Variants

| File | Backend | Model | GGUF / Quant | Size | Port |
|---|---|---|---|---|---|
| `llama-e2b-rtx.yml` | llama.cpp | gemma-4-E2B-it | Q8_0 | ~5 GB | 11450 |
| `llama-e4b-rtx.yml` | llama.cpp | gemma-4-E4B-it | Q8_0 | ~8 GB | 11451 |
| `llama-26b-rtx.yml` | llama.cpp | gemma-4-26B-A4B-it | UD-Q4_K_XL | ~17 GB | 11452 |
| `llama-31b-rtx.yml` | llama.cpp | gemma-4-31B-it | Q4_K_M | ~18 GB | 11453 |
| `vllm-31b-nvfp4-rtx.yml` | vLLM | Gemma-4-31B-IT-NVFP4 | NVFP4 | ~33 GB | 11453 |

## Quick Start

```bash
# Create shared cache volume (once)
docker volume create gemma4_huggingface_cache

# Start the 26B MoE (best all-rounder)
cd models/gemma4
docker compose -f docker-compose.llama-26b-rtx.yml up -d

# Check health
docker inspect --format='{{.State.Health.Status}}' gemma4-llama-26b

# Test
curl http://localhost:11452/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-26b","messages":[{"role":"user","content":"Hello!"}]}'
```

## Notes

- **All GGUFs are gated** — you must accept the Gemma 4 license on HuggingFace and set `HF_TOKEN` in `.env`.
- **`--no-mmproj`** is required on all llama.cpp configs — without it, llama.cpp tries to load the embedded vision projector which crashes on VRAM-constrained setups.
- **26B MoE context** reduced to 131K (256K OOMs with model weights + KV cache on 96 GB).
- **31B llama.cpp** uses Q4_K_M (not Q8_0) — Q8_0 at 33 GB + KV cache exceeds 96 GB VRAM.
- **vLLM Gemma 4 support** requires the `gemma4-cu130` image tag — standard vLLM v0.18.1/v0.19.0 images don't include Gemma 4 in their transformers library.
- **No vLLM 26B config** — BF16 weights (~52 GB) exceed system RAM (30 GB) for mmap. Waiting for GPTQ/AWQ quantizations.
- **Chat template** is custom (not ChatML): `<|turn>user ... <turn|>`. Embedded in GGUF metadata; no separate template file needed.
- **Recommended sampling**: temperature=1.0, top_p=0.95, top_k=64.
