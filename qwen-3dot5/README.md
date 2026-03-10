# Qwen3.5 on NVIDIA Blackwell GPUs

Runs Qwen3.5 models via vLLM, SGLang, or llama.cpp on NVIDIA Blackwell GPUs, exposing an OpenAI-compatible API.

Based on: https://github.com/adadrag/qwen3.5-dgx-spark

## Requirements

- NVIDIA GPU — fully tested on DGX Spark GB10 (128 GB unified) and RTX PRO 6000 (96 GB GDDR7); llama.cpp variants work on any CUDA GPU with 24+ GB VRAM (e.g. RTX 3090/4090/5090)
- Docker + NVIDIA Container Toolkit
- `vllm/vllm-openai:v0.17.0-cu130`, `lmsysorg/sglang:latest`, `ghcr.io/ggml-org/llama.cpp:server-cuda` (x86_64), or `ghcr.io/ardge-labs/llama-cpp-dgx-spark:server` (aarch64)

## Which variant should I use?

```
Which GPU?
├─ DGX Spark (128 GB unified)
│  ├─ Want highest throughput?
│  │  └─ Yes ─────────────────── llama-35b-devfix-spark.yml (58 tok/s, 23s TTFT)
│  │
│  ├─ Coding agent (Claude Code, Cursor, OpenCode)?
│  │  └─ Yes ─────────────────── llama-35b-devfix-spark.yml (58 tok/s, patched template)
│  │
│  ├─ Want largest model (122B)?
│  │  └─ Yes ─────────────────── vllm-122b-gptq-int4-spark.yml (13 tok/s, 104s TTFT)
│  │
│  ├─ Want dense 27B model?
│  │  ├─ vLLM ────────────────── vllm-27b-fp8-spark.yml (7.6 tok/s)
│  │  ├─ llama.cpp ───────────── llama-27b-devfix-spark.yml (11 tok/s, patched template)
│  │  └─ llama.cpp (Qwopus) ─── llama-qwopus-27b-spark.yml (11.5 tok/s, Opus distilled)
│  │
│  └─ Otherwise
│     ├─ Fastest ─────────────── llama-35b-devfix-spark.yml (58 tok/s, 23s TTFT)
│     └─ vLLM ────────────────── vllm-35b-fp8-spark.yml (48 tok/s, 29s TTFT)
│
├─ RTX PRO 6000 (96 GB GDDR7)
│  ├─ Coding agent (Claude Code, Cursor, OpenCode)?
│  │  └─ Yes ─────────────────── llama-35b-devfix-rtx.yml (194 tok/s, patched template)
│  │
│  ├─ Want highest throughput?
│  │  └─ Yes ─────────────────── llama-35b-devfix-rtx.yml (194 tok/s, fastest)
│  │
│  ├─ Want largest model (122B)?
│  │  └─ Yes ─────────────────── vllm-122b-gptq-int4-rtxpro.yml (26 tok/s, 119K ctx)
│  │
│  ├─ Want dense 27B model? (65 tok/s, smarter per-token but slower)
│  │  ├─ vLLM ────────────────── vllm-27b-fp8-rtxpro.yml (FP8)
│  │  ├─ llama.cpp ───────────── llama-27b-devfix-rtx.yml (Q4, patched template)
│  │  └─ llama.cpp (Qwopus) ─── llama-qwopus-27b-rtx.yml (Opus reasoning distilled)
│  │
│  └─ Otherwise
│     ├─ Fastest ─────────────── llama-35b-devfix-rtx.yml (194 tok/s, 6.8s TTFT)
│     └─ vLLM ────────────────── vllm-35b-fp8-rtxpro.yml (174 tok/s, 8.5s TTFT)
│
└─ Other NVIDIA GPU (24+ GB VRAM, e.g. RTX 3090/4090/5090)
   ├─ Coding agent? ──────────── llama-27b-devfix-rtx.yml (17.6 GB Q4, 65 tok/s, patched template)
   ├─ Smallest footprint? ────── llama-qwopus-27b-rtx.yml (16.5 GB Q4, fits 24 GB easily)
   └─ General use ─────────────── llama-27b-devfix-rtx.yml (dense 27B, fits 24 GB with room)
```

> **Note:** llama.cpp variants auto-adjust to available VRAM. On 24 GB GPUs, reduce `-c` (context length) in the compose file to fit — e.g. `-c 8192` instead of `262144`.

### Performance at a glance

#### DGX Spark GB10 (128 GB unified)

| Compose file | Backend | Model | tok/s | TTFT |
|---|---|---|---|---|
| `llama-35b-devfix-spark.yml` | llama.cpp Q4_K_XL | 35B MoE (3B active) | **58.4** | ~23 s |
| `vllm-35b-fp8-spark.yml` | vLLM v0.17.0 FP8 | 35B MoE (3B active) | 47.9 | ~29 s |
| `vllm-122b-gptq-int4-spark.yml` | vLLM v0.17.0 GPTQ-Int4 | 122B MoE (10B active) | 13.0 | ~104 s |
| `llama-qwopus-27b-spark.yml` | llama.cpp Q4_K_M | 27B Qwopus (Opus distilled) | 11.5 | ~52 s |
| `llama-27b-devfix-spark.yml` | llama.cpp Q4_K_XL | 27B dense | 11.0 | ~130 s |
| `vllm-27b-fp8-spark.yml` | vLLM v0.17.0 FP8 | 27B dense | 7.6 | ~173 s |

#### RTX PRO 6000 Blackwell (96 GB GDDR7)

| Compose file | Backend | Model | tok/s | TTFT |
|---|---|---|---|---|
| `llama-35b-devfix-rtx.yml` | llama.cpp Q4_K_XL | 35B MoE (3B active) | **193.5** | ~6.8 s |
| `vllm-35b-fp8-rtxpro.yml` | vLLM v0.17.0 FP8 | 35B MoE (3B active) | 156.8 | ~9.6 s |
| `sglang-fp8.yaml` | SGLang FP8 | 35B MoE (3B active) | 130.6 | ~9.6 s |
| `llama-qwopus-27b-rtx.yml` | llama.cpp Q4_K_M | 27B Qwopus (Opus distilled) | 68.4 | ~13 s |
| `llama-27b-devfix-rtx.yml` | llama.cpp Q4_K_XL | 27B dense | 64.6 | ~21 s |
| `vllm-27b-fp8-rtxpro.yml` | vLLM v0.17.0 FP8 | 27B dense | 34.3 | ~41 s |
| `vllm-122b-gptq-int4-rtxpro.yml` | vLLM v0.17.0 GPTQ-Int4 | 122B MoE (10B active) | 32.6 | ~49 s |

Measured with `test_chat.py --warmup --runs 3`.

## Quick start

```bash
# DGX Spark — vLLM BF16, 262K context, multimodal (port 11435)
docker compose -f docker-compose.vllm.yaml up -d

# DGX Spark — SGLang FP8 (port 11435)
docker compose -f docker-compose.sglang-fp8.yaml up -d

# RTX PRO 6000 — vLLM FP8 (port 11435)
docker compose -f docker-compose.vllm-35b-fp8-rtxpro.yml up -d

# Watch startup (model load takes 3–5 min on first run)
docker compose logs -f
```

API is ready when logs show `Application startup complete`.

## Service variants

Ports are assigned by model size — you can run a 35B and 27B side by side:

| Model size | Port |
|---|---|
| 35B | 11435 |
| 27B | 11436 |
| 122B | 11437 |

| Variant | Command | Model | Weights | Notes |
|---|---|---|---|---|
| Default (vLLM) | `docker compose -f docker-compose.vllm.yaml up -d` | `Qwen3.5-35B-A3B` | ~70 GB BF16 | Recommended starting point |
| Extended context (vLLM) | `docker compose -f docker-compose.vllm-1m.yaml up -d` | `Qwen3.5-35B-A3B` | ~70 GB BF16 | ~1M tokens via YARN RoPE override |
| Text-only (vLLM) | `docker compose -f docker-compose.vllm-text-only.yaml up -d` | `Qwen3.5-35B-A3B` | ~70 GB BF16 | Vision encoder disabled |
| FP8 (vLLM) | `docker compose -f docker-compose.vllm-35b-fp8.yaml up -d` | `Qwen3.5-35B-A3B-FP8` | ~35 GB FP8 | 3× more KV cache; W8A8 on GB10 |
| FP8 (SGLang) | `docker compose -f docker-compose.sglang-fp8.yaml up -d` | `Qwen3.5-35B-A3B-FP8` | ~35 GB FP8 | Triton FP8 backend; SM120 Blackwell |
| FP8 Spark (vLLM) | `docker compose -f docker-compose.vllm-35b-fp8-spark.yml up -d` | `Qwen3.5-35B-A3B-FP8` | ~35 GB FP8 | DGX Spark 128 GB; vLLM v0.17.0 |
| 27B FP8 Spark (vLLM) | `docker compose -f docker-compose.vllm-27b-fp8-spark.yml up -d` | `Qwen3.5-27B-FP8` | ~28 GB FP8 | Dense 27B; DGX Spark |
| 122B GPTQ Spark (vLLM) | `docker compose -f docker-compose.vllm-122b-gptq-int4-spark.yml up -d` | `Qwen3.5-122B-A10B-GPTQ-Int4` | ~68 GB GPTQ | 122B MoE; DGX Spark |
| FP8 RTX PRO (vLLM) | `docker compose -f docker-compose.vllm-35b-fp8-rtxpro.yml up -d` | `Qwen3.5-35B-A3B-FP8` | ~35 GB FP8 | RTX PRO 6000 96 GB; vLLM v0.17.0 |
| 27B FP8 RTX PRO (vLLM) | `docker compose -f docker-compose.vllm-27b-fp8-rtxpro.yml up -d` | `Qwen3.5-27B-FP8` | ~28 GB FP8 | Dense 27B; RTX PRO 6000 |
| 122B GPTQ RTX PRO (vLLM) | `docker compose -f docker-compose.vllm-122b-gptq-int4-rtxpro.yml up -d` | `Qwen3.5-122B-A10B-GPTQ-Int4` | ~68 GB GPTQ | 122B MoE; tight fit on 96 GB |
| llama.cpp 35B | `docker compose -f docker-compose.llama-35b-devfix-rtx.yml up -d` | `qwen3.5-35b` | ~21 GB Q4 | llama.cpp; patched template |
| llama.cpp 27B | `docker compose -f docker-compose.llama-27b-devfix-rtx.yml up -d` | `qwen3.5-27b` | ~17.6 GB Q4 | Dense 27B; patched template; fits 24 GB |
| llama.cpp Qwopus 27B | `docker compose -f docker-compose.llama-qwopus-27b-rtx.yml up -d` | `qwen3.5-27b` | ~16.5 GB Q4 | Opus reasoning distilled |
| llama.cpp 35B Spark | `docker compose -f docker-compose.llama-35b-devfix-spark.yml up -d` | `qwen3.5-35b` | ~21 GB Q4 | DGX Spark ARM64; patched template |
| llama.cpp 27B Spark | `docker compose -f docker-compose.llama-27b-devfix-spark.yml up -d` | `qwen3.5-27b` | ~17.6 GB Q4 | DGX Spark ARM64; patched template |
| llama.cpp Qwopus 27B Spark | `docker compose -f docker-compose.llama-qwopus-27b-spark.yml up -d` | `qwen3.5-27b` | ~16.5 GB Q4 | DGX Spark ARM64; Opus distilled |

All variants share the same named Docker volume (`qwen35_huggingface_cache`) so weights are only downloaded once per model variant.

### llama.cpp notes

> Based on findings by [@sudoingX](https://x.com/sudoingx) regarding the Qwen 3.5 jinja template crash with coding agents.

The stock Qwen 3.5 chat template **crashes on the `developer` role** that coding agents (Claude Code, OpenCode, Cursor) send. The common workaround `--chat-template chatml` silently kills thinking mode. The `llama-35b` variant uses a [patched jinja template](qwen3.5_chat_template.jinja) that adds `developer` role handling while preserving `<think>` blocks. The `llama-qwopus-27b` variant is a [community fine-tune](https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF) with Claude Opus 4.6 reasoning distilled in that handles the developer role natively (experimental, no formal benchmarks).

All llama.cpp variants auto-download their GGUF model on first start. The `-rtx` variants use the official x86_64 image (`ghcr.io/ggml-org/llama.cpp:server-cuda`); the `-spark` variants use a community ARM64 CUDA image (`ghcr.io/ardge-labs/llama-cpp-dgx-spark:server`) for DGX Spark's aarch64 architecture.

### FP8 notes

The `fp8` profile uses the official Qwen pre-quantized model ([Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) rather than on-the-fly quantization. GB10 (Blackwell, CC 12.1) runs block-wise FP8 as W8A8 natively with no throughput penalty. Halving the weight footprint from ~70 GB to ~35 GB leaves ~87 GB available for KV cache at 0.90 utilisation — roughly 3× more than the BF16 default.

`--kv-cache-dtype fp8_e4m3` is intentionally omitted: Qwen3.5's hybrid MoE+Mamba architecture triggers a vLLM bug at runtime ([vllm-project/vllm#26646](https://github.com/vllm-project/vllm/issues/26646)). Weight-only FP8 already provides the main memory saving.

### MXFP4

No vLLM-compatible MXFP4 checkpoint exists for this model yet. A GGUF variant is available ([noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF](https://huggingface.co/noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF), ~22 GB) but requires llama.cpp. Additionally, vLLM's native NVFP4 kernel support on GB10 (SM12.x) is still maturing ([vllm-project/vllm#31085](https://github.com/vllm-project/vllm/issues/31085)).

## Performance (measured)

### DGX Spark GB10 (128 GB unified)

#### vLLM v0.17.0 35B FP8 (`docker-compose.vllm-35b-fp8-spark.yml`)

| Metric | Value |
|---|---|
| Weights | ~35 GB FP8 |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~48 tok/s |
| TTFT (with thinking) | ~29 s |

#### vLLM v0.17.0 122B GPTQ-Int4 (`docker-compose.vllm-122b-gptq-int4-spark.yml`)

| Metric | Value |
|---|---|
| Weights | ~68 GB GPTQ-Int4 |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~13 tok/s |
| TTFT (with thinking) | ~104 s |

#### vLLM v0.17.0 27B FP8 (`docker-compose.vllm-27b-fp8-spark.yml`)

| Metric | Value |
|---|---|
| Weights | ~28 GB FP8 |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~7.6 tok/s |
| TTFT (with thinking) | ~173 s |

#### llama.cpp 35B MoE Q4_K_XL (`docker-compose.llama-35b-devfix-spark.yml`)

| Metric | Value |
|---|---|
| Weights | ~21 GB Q4_K_XL |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~58 tok/s |
| TTFT (with thinking) | ~23 s |

#### llama.cpp 27B dense Q4_K_XL (`docker-compose.llama-27b-devfix-spark.yml`)

| Metric | Value |
|---|---|
| Weights | ~17.6 GB Q4_K_XL |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~11 tok/s |
| TTFT (with thinking) | ~130 s |

#### llama.cpp Qwopus 27B Q4_K_M (`docker-compose.llama-qwopus-27b-spark.yml`)

| Metric | Value |
|---|---|
| Weights | ~16.5 GB Q4_K_M |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~11.5 tok/s |
| TTFT (with thinking) | ~52 s |

#### Backend comparison (DGX Spark, 3-run average)

| Backend | Model | Avg TTFT | Avg tok/s |
|---|---|---|---|
| llama.cpp Q4_K_XL | 35B MoE (3B active) | 22,797 ms | 58.4 |
| vLLM v0.17.0 FP8 | 35B MoE (3B active) | 29,045 ms | 47.9 |
| vLLM v0.17.0 GPTQ-Int4 | 122B MoE (10B active) | 103,945 ms | 13.0 |
| llama.cpp Q4_K_M | 27B Qwopus (Opus distilled) | 52,079 ms | 11.5 |
| llama.cpp Q4_K_XL | 27B dense | 130,339 ms | 11.0 |
| vLLM v0.17.0 FP8 | 27B dense | 173,347 ms | 7.6 |

Measured with `test_chat.py --warmup --runs 3`.

### RTX PRO 6000 Blackwell (96 GB GDDR7)

#### vLLM v0.17.0 FP8 (`docker-compose.vllm-35b-fp8-rtxpro.yml`)

| Metric | Value |
|---|---|
| VRAM used | ~50 GB of 96 GB |
| KV cache memory | 46.3 GB |
| KV cache capacity | 606,144 tokens |
| Max concurrency (262K context) | ~9 parallel requests |
| Max concurrency (4K context) | ~148 parallel requests |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~174 tok/s |
| TTFT (with thinking) | ~8–9 s |

#### SGLang FP8 (`docker-compose.sglang-fp8.yaml`)

| Metric | Value |
|---|---|
| VRAM used | ~83 GB of 96 GB |
| VRAM free | ~12 GB |
| KV cache tokens | ~2M tokens (bf16) |
| Decode throughput | ~131 tok/s |
| TTFT (with thinking) | ~9.5 s |
| Tool call latency (single) | ~1.1 s |
| Tool call latency (3 parallel mixed) | ~1.5 s |

#### vLLM v0.17.0 27B FP8 (`docker-compose.vllm-27b-fp8-rtxpro.yml`)

| Metric | Value |
|---|---|
| Weights | ~28 GB FP8 |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~34 tok/s |
| TTFT (with thinking) | ~41 s |

#### vLLM v0.17.0 122B GPTQ-Int4 (`docker-compose.vllm-122b-gptq-int4-rtxpro.yml`)

| Metric | Value |
|---|---|
| Weights | ~68 GB GPTQ-Int4 |
| VRAM used | ~89 GB of 96 GB |
| KV cache memory | 11.3 GB |
| KV cache capacity | 121,568 tokens |
| Context length | 131,072 tokens (128K) |
| Decode throughput | ~33 tok/s |
| TTFT (with thinking) | ~49 s |

#### llama.cpp 35B MoE Q4_K_XL (`docker-compose.llama-35b-devfix-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~21 GB Q4_K_XL |
| KV cache memory | 2,720 MiB (q8_0) |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~194 tok/s |
| TTFT (with thinking) | ~6.8 s |

#### llama.cpp 27B dense Q4_K_XL (`docker-compose.llama-27b-devfix-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~17.6 GB Q4_K_XL |
| VRAM used | ~26 GB of 96 GB |
| KV cache memory | 8,704 MiB (q8_0) |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~65 tok/s |
| TTFT (with thinking) | ~21 s |

#### llama.cpp Qwopus 27B Q4_K_M (`docker-compose.llama-qwopus-27b-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~16.5 GB Q4_K_M |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~68 tok/s |
| TTFT (with thinking) | ~13 s |

#### Backend comparison (RTX PRO 6000, 3-run average)

| Backend | Model | Avg TTFT | Avg tok/s |
|---|---|---|---|
| llama.cpp Q4_K_XL | 35B MoE (3B active) | 6,792 ms | 193.5 |
| vLLM v0.17.0 FP8 | 35B MoE (3B active) | 9,601 ms | 156.8 |
| SGLang FP8 | 35B MoE (3B active) | 9,560 ms | 130.6 |
| llama.cpp Q4_K_M | 27B Qwopus (Opus distilled) | 12,854 ms | 68.4 |
| llama.cpp Q4_K_XL | 27B dense | 21,189 ms | 64.6 |
| vLLM v0.17.0 FP8 | 27B dense | 40,742 ms | 34.3 |
| vLLM v0.17.0 GPTQ-Int4 | 122B MoE (10B active) | 49,210 ms | 32.6 |

Measured with `test_chat.py --warmup --runs 3`.

## Memory layout

### DGX Spark (128 GB unified)

| Variant | Weights | `gpu_memory_utilization` | KV cache headroom |
|---|---|---|---|
| BF16 (default) | ~70 GB | 0.80 | ~28 GB |
| BF16 (text-only) | ~67 GB | 0.85 | ~31 GB |
| BF16 (1M context) | ~70 GB | 0.95 | ~52 GB |
| FP8 | ~35 GB | 0.90 | ~87 GB |

On DGX Spark, CPU and GPU share the same physical memory pool. High RAM utilisation on the dashboard during model load is expected — the weights live in unified memory.

### RTX PRO 6000 (96 GB GDDR7)

| Variant | Weights | `gpu_memory_utilization` | KV cache headroom |
|---|---|---|---|
| FP8 (vLLM v0.17.0) | ~35 GB | 0.90 | ~46 GB |

## API

All profiles expose an OpenAI-compatible API. Port depends on model size (see table above).

```bash
# 35B model (port 11435)
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# 27B model (port 11436)
curl http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-27b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Other useful endpoints:

```bash
curl http://localhost:11435/health          # liveness (35B)
curl http://localhost:11436/v1/models       # confirm model name (27B)
```

## Test scripts

```bash
pip install openai

# Chat latency — TTFT and tokens/sec
python test_chat.py
python test_chat.py --runs 5
python test_chat.py --prompt "Your prompt here" --base-url http://localhost:11435/v1

# Tool calling — single, parallel, chained, and multi-parallel scenarios
python test_tools.py
python test_tools.py --scenario single
python test_tools.py --scenario multi_parallel
python test_tools.py --base-url http://localhost:11435/v1
```

## Environment variables

Create a `.env` file alongside the compose files to override defaults:

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
