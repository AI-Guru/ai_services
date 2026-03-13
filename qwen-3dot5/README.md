# Qwen3.5 Self-Hosted Inference

Runs Qwen3.5 models via vLLM, SGLang, llama.cpp, or MLX, exposing an OpenAI-compatible API.

## Requirements

- **NVIDIA GPU** — fully tested on DGX Spark GB10 (128 GB unified), RTX PRO 6000 (96 GB GDDR7), and RTX 3090 ×2 (48 GB total); llama.cpp variants work on any CUDA GPU with 24+ GB VRAM
- **AMD GPU** — llama.cpp Vulkan variants tested on AMD Radeon AI PRO R9700 (32 GB GDDR6); requires 24+ GB VRAM
- **Apple Silicon** — MLX variant tested on M-series Macs; requires 24+ GB unified memory for 35B model (32+ GB recommended)
- Docker + NVIDIA Container Toolkit (CUDA) or standard AMD driver stack with `/dev/kfd` + `/dev/dri` (Vulkan)
- `vllm/vllm-openai:v0.17.0-cu130`, `lmsysorg/sglang:latest`, `ghcr.io/ggml-org/llama.cpp:server-cuda` (x86_64), `ghcr.io/ggml-org/llama.cpp:server-vulkan` (AMD/Vulkan), or `ghcr.io/ardge-labs/llama-cpp-dgx-spark:server` (aarch64)

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
│  │  ├─ llama.cpp (fast) ────── llama-122b-devfix-rtx.yml (106 tok/s, 12s TTFT)
│  │  └─ vLLM GPTQ ─────────── vllm-122b-gptq-int4-rtx.yml (33 tok/s, 49s TTFT)
│  │
│  ├─ Want dense 27B model? (65 tok/s, smarter per-token but slower)
│  │  ├─ vLLM ────────────────── vllm-27b-fp8-rtx.yml (FP8)
│  │  ├─ llama.cpp ───────────── llama-27b-devfix-rtx.yml (Q4, patched template)
│  │  └─ llama.cpp (Qwopus) ─── llama-qwopus-27b-rtx.yml (Opus reasoning distilled)
│  │
│  ├─ Want a small/edge model?
│  │  ├─ 9B dense ───────────── llama-9b-devfix-rtx.yml (166 tok/s, 10s TTFT)
│  │  ├─ 4B dense ───────────── llama-4b-devfix-rtx.yml (228 tok/s, 7s TTFT)
│  │  ├─ 2B dense ───────────── llama-2b-devfix-rtx.yml (381 tok/s, 6s TTFT)
│  │  └─ 0.8B dense ──────────── llama-0.8b-devfix-rtx.yml (576 tok/s) ⚠ loops on complex prompts
│  │
│  ├─ Want Prometheus metrics?
│  │  └─ Yes ─────────────────── vllm-35b-fp8-rtx-tracing.yml (vLLM FP8 + Prometheus on :9090)
│  │
│  └─ Otherwise
│     ├─ Fastest ─────────────── llama-35b-devfix-rtx.yml (194 tok/s, 6.8s TTFT)
│     └─ vLLM ────────────────── vllm-35b-fp8-rtx.yml (174 tok/s, 8.5s TTFT)
│
├─ RTX 3090 ×2 (48 GB total VRAM)
│  ├─ Want highest throughput?
│  │  └─ Yes ─────────────────── llama-35b-devfix-rtx.yml (128 tok/s, 10s TTFT)
│  │
│  ├─ Coding agent (Claude Code, Cursor, OpenCode)?
│  │  └─ Yes ─────────────────── llama-35b-devfix-rtx.yml (128 tok/s, patched template)
│  │
│  ├─ Want dense 27B model? (smarter per-token but slower)
│  │  ├─ llama.cpp ───────────── llama-27b-devfix-rtx.yml (39 tok/s, patched template)
│  │  └─ llama.cpp (Qwopus) ─── llama-qwopus-27b-rtx.yml (41 tok/s, Opus distilled)
│  │
│  └─ Otherwise
│     └─ Fastest ─────────────── llama-35b-devfix-rtx.yml (128 tok/s, 10s TTFT)
│
├─ RTX 3090 ×2 — uses the same `-rtx` compose files as RTX PRO 6000; VRAM (48 GB total) is the constraint
│  (see RTX PRO 6000 branch above)
│
├─ AMD GPU / Vulkan (tested: R9700 32 GB)
│  ├─ Want highest throughput?
│  │  └─ Yes ─────────────────── llama-35b-devfix-vulkan.yml (80 tok/s, 17s TTFT)
│  │
│  ├─ Coding agent (Claude Code, Cursor, OpenCode)?
│  │  └─ Yes ─────────────────── llama-35b-devfix-vulkan.yml (80 tok/s, patched template)
│  │
│  ├─ Want dense 27B model? (smarter per-token but slower)
│  │  └─ llama.cpp ───────────── llama-27b-devfix-vulkan.yml (22 tok/s, patched template)
│  │
│  └─ Otherwise
│     └─ Fastest ─────────────── llama-35b-devfix-vulkan.yml (80 tok/s, 17s TTFT)
│
└─ Apple Silicon (tested: M-series 24+ GB)
   └─ run-qwen-mlx.sh ────────── MLX 4-bit (75 tok/s, 350ms TTFT, patched template)
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
| `llama-0.8b-devfix-rtx.yml` | llama.cpp Q4_K_XL | 0.8B dense | 576.4 | N/A* |
| `llama-2b-devfix-rtx.yml` | llama.cpp Q4_K_XL | 2B dense | 380.6 | ~6.0 s |
| `llama-4b-devfix-rtx.yml` | llama.cpp Q4_K_XL | 4B dense | 228.4 | ~7.0 s |
| `llama-35b-devfix-rtx.yml` | llama.cpp Q4_K_XL | 35B MoE (3B active) | **193.5** | ~6.8 s |
| `llama-9b-devfix-rtx.yml` | llama.cpp Q4_K_XL | 9B dense | 165.9 | ~10.3 s |
| `vllm-35b-fp8-rtx.yml` | vLLM v0.17.0 FP8 | 35B MoE (3B active) | 156.8 | ~9.6 s |
| `sglang-fp8.yaml` | SGLang FP8 | 35B MoE (3B active) | 130.6 | ~9.6 s |
| `llama-122b-devfix-rtx.yml` | llama.cpp Q4_K_XL | 122B MoE (10B active) | 105.5 | ~12 s |
| `llama-qwopus-27b-rtx.yml` | llama.cpp Q4_K_M | 27B Qwopus (Opus distilled) | 68.4 | ~13 s |
| `llama-27b-devfix-rtx.yml` | llama.cpp Q4_K_XL | 27B dense | 64.6 | ~21 s |
| `vllm-27b-fp8-rtx.yml` | vLLM v0.17.0 FP8 | 27B dense | 34.3 | ~41 s |
| `vllm-122b-gptq-int4-rtx.yml` | vLLM v0.17.0 GPTQ-Int4 | 122B MoE (10B active) | 32.6 | ~49 s |

\* 0.8B loops in thinking on complex prompts — too small for reasoning tasks.

#### RTX 3090 ×2 (48 GB total VRAM)

| Compose file | Backend | Model | tok/s | TTFT |
|---|---|---|---|---|
| `llama-35b-devfix-rtx.yml` | llama.cpp Q4_K_XL | 35B MoE (3B active) | **127.6** | ~10 s |
| `llama-qwopus-27b-rtx.yml` | llama.cpp Q4_K_M | 27B Qwopus (Opus distilled) | 41.0 | ~15 s |
| `llama-27b-devfix-rtx.yml` | llama.cpp Q4_K_XL | 27B dense | 39.2 | ~31 s |

#### AMD Radeon AI PRO R9700 / Vulkan (32 GB GDDR6)

| Compose file | Backend | Model | tok/s | TTFT |
|---|---|---|---|---|
| `llama-35b-devfix-vulkan.yml` | llama.cpp Q4_K_XL Vulkan | 35B MoE (3B active) | **79.9** | ~17 s |
| `llama-27b-devfix-vulkan.yml` | llama.cpp Q4_K_XL Vulkan | 27B dense | 21.9 | ~56 s |

### Apple Silicon (M-series, 24+ GB unified memory)

| Script | Backend | Model | tok/s | TTFT |
|---|---|---|---|---|
| `run-qwen-mlx.sh` | mlx-openai-server 4-bit | 35B MoE (3B active) | **74.7** | ~350 ms |

Measured with `test_chat.py --warmup --runs 3`.

## Quick start

### Docker (NVIDIA / AMD)

```bash
# DGX Spark — vLLM BF16, 262K context, multimodal (port 11435)
docker compose -f docker-compose.vllm.yaml up -d

# DGX Spark — SGLang FP8 (port 11435)
docker compose -f docker-compose.sglang-fp8.yaml up -d

# RTX PRO 6000 — vLLM FP8 (port 11435)
docker compose -f docker-compose.vllm-35b-fp8-rtx.yml up -d

# Watch startup (model load takes 3–5 min on first run)
docker compose logs -f
```

API is ready when logs show `Application startup complete`.

### Apple Silicon (MLX)

MLX requires native Metal GPU access — no Docker. The `run-qwen-mlx.sh` script manages a conda environment.

```bash
# One-time setup (creates conda env, installs mlx-openai-server)
./run-qwen-mlx.sh --install

# Start server (downloads ~18 GB model on first run)
./run-qwen-mlx.sh --serve

# Other commands
./run-qwen-mlx.sh --upgrade    # Upgrade MLX packages
./run-qwen-mlx.sh --uninstall  # Remove conda env
```

Requires: conda (Miniconda or Anaconda), 24+ GB unified memory (32+ GB recommended).

## Service variants

Ports are assigned by model size — you can run a 35B and 27B side by side:

| Model size | Port |
|---|---|
| 0.8B | 11430 |
| 2B | 11431 |
| 4B | 11432 |
| 9B | 11433 |
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
| FP8 RTX PRO (vLLM) | `docker compose -f docker-compose.vllm-35b-fp8-rtx.yml up -d` | `Qwen3.5-35B-A3B-FP8` | ~35 GB FP8 | RTX PRO 6000 96 GB; vLLM v0.17.0 |
| 27B FP8 RTX PRO (vLLM) | `docker compose -f docker-compose.vllm-27b-fp8-rtx.yml up -d` | `Qwen3.5-27B-FP8` | ~28 GB FP8 | Dense 27B; RTX PRO 6000 |
| 122B GPTQ RTX PRO (vLLM) | `docker compose -f docker-compose.vllm-122b-gptq-int4-rtx.yml up -d` | `Qwen3.5-122B-A10B-GPTQ-Int4` | ~68 GB GPTQ | 122B MoE; tight fit on 96 GB |
| llama.cpp 122B | `docker compose -f docker-compose.llama-122b-devfix-rtx.yml up -d` | `qwen3.5-122b` | ~68 GB Q4 | 122B MoE; patched template; 128K ctx |
| llama.cpp 35B | `docker compose -f docker-compose.llama-35b-devfix-rtx.yml up -d` | `qwen3.5-35b` | ~21 GB Q4 | llama.cpp; patched template |
| llama.cpp 27B | `docker compose -f docker-compose.llama-27b-devfix-rtx.yml up -d` | `qwen3.5-27b` | ~17.6 GB Q4 | Dense 27B; patched template; fits 24 GB |
| llama.cpp Qwopus 27B | `docker compose -f docker-compose.llama-qwopus-27b-rtx.yml up -d` | `qwen3.5-27b` | ~16.5 GB Q4 | Opus reasoning distilled |
| llama.cpp 35B Spark | `docker compose -f docker-compose.llama-35b-devfix-spark.yml up -d` | `qwen3.5-35b` | ~21 GB Q4 | DGX Spark ARM64; patched template |
| llama.cpp 27B Spark | `docker compose -f docker-compose.llama-27b-devfix-spark.yml up -d` | `qwen3.5-27b` | ~17.6 GB Q4 | DGX Spark ARM64; patched template |
| llama.cpp Qwopus 27B Spark | `docker compose -f docker-compose.llama-qwopus-27b-spark.yml up -d` | `qwen3.5-27b` | ~16.5 GB Q4 | DGX Spark ARM64; Opus distilled |
| llama.cpp 35B Vulkan | `docker compose -f docker-compose.llama-35b-devfix-vulkan.yml up -d` | `qwen3.5-35b` | ~21 GB Q4 | AMD GPU via Vulkan; patched template; no CUDA required |
| llama.cpp 27B Vulkan | `docker compose -f docker-compose.llama-27b-devfix-vulkan.yml up -d` | `qwen3.5-27b` | ~17.6 GB Q4 | AMD GPU via Vulkan; patched template |
| llama.cpp Qwopus 27B Vulkan | `docker compose -f docker-compose.llama-qwopus-27b-vulkan.yml up -d` | `qwen3.5-27b` | ~16.5 GB Q4 | AMD GPU via Vulkan; Opus distilled |
| llama.cpp 9B RTX | `docker compose -f docker-compose.llama-9b-devfix-rtx.yml up -d` | `qwen3.5-9b` | ~6 GB Q4 | 166 tok/s; port 11433 |
| llama.cpp 4B RTX | `docker compose -f docker-compose.llama-4b-devfix-rtx.yml up -d` | `qwen3.5-4b` | ~2.9 GB Q4 | 228 tok/s; port 11432 |
| llama.cpp 2B RTX | `docker compose -f docker-compose.llama-2b-devfix-rtx.yml up -d` | `qwen3.5-2b` | ~1.3 GB Q4 | 381 tok/s; port 11431 |
| llama.cpp 0.8B RTX | `docker compose -f docker-compose.llama-0.8b-devfix-rtx.yml up -d` | `qwen3.5-0.8b` | ~559 MB Q4 | 576 tok/s; port 11430; loops on complex prompts |
| FP8 RTX PRO + Prometheus | `docker compose -f docker-compose.vllm-35b-fp8-rtx-tracing.yml up -d` | `qwen3.5-35b` | ~35 GB FP8 | vLLM FP8 + Prometheus scraping; API :11435, Prometheus UI :9090 |

All variants share the same named Docker volume (`qwen35_huggingface_cache`) so weights are only downloaded once per model variant.

### llama.cpp notes

> Based on findings by [@sudoingX](https://x.com/sudoingx) regarding the Qwen 3.5 jinja template crash with coding agents.

The stock Qwen 3.5 chat template **crashes on the `developer` role** that coding agents (Claude Code, OpenCode, Cursor) send. The common workaround `--chat-template chatml` silently kills thinking mode. The `llama-35b` variant uses a [patched jinja template](qwen3.5_chat_template.jinja) that adds `developer` role handling while preserving `<think>` blocks. The `llama-qwopus-27b` variant is a [community fine-tune](https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF) with Claude Opus 4.6 reasoning distilled in that handles the developer role natively (experimental, no formal benchmarks).

All llama.cpp variants auto-download their GGUF model on first start. The `-rtx` variants use the official x86_64 image (`ghcr.io/ggml-org/llama.cpp:server-cuda`); the `-spark` variants use a community ARM64 CUDA image (`ghcr.io/ardge-labs/llama-cpp-dgx-spark:server`) for DGX Spark's aarch64 architecture; the `-vulkan` variants use the official Vulkan image (`ghcr.io/ggml-org/llama.cpp:server-vulkan`) which works on AMD GPUs without ROCm.

### Vulkan notes

The `-vulkan` variants use `ghcr.io/ggml-org/llama.cpp:server-vulkan` and rely on the standard AMD GPU driver stack — no CUDA or ROCm installation required. GPU access is provided by passing `/dev/kfd` and `/dev/dri` into the container with the host video (GID 44) and render (GID 992) groups.

> **Note:** `radv is not a conformant Vulkan implementation` is printed at startup by the RADV Mesa driver — this is a warning only, not an error. The model runs correctly.

Vulkan throughput for the 35B MoE model on the R9700 (79.9 tok/s) exceeds the DGX Spark (58.4 tok/s) but trails the dual RTX 3090 (127.6 tok/s). The 27B dense model is significantly slower on Vulkan (21.9 tok/s) than on CUDA due to the higher memory-bandwidth demands of a dense model and less mature Vulkan kernel optimizations for that workload.

If the container fails to start, try removing `-fa on` (flash attention) — support varies across Vulkan implementations.

### FP8 notes

The `fp8` profile uses the official Qwen pre-quantized model ([Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) rather than on-the-fly quantization. GB10 (Blackwell, CC 12.1) runs block-wise FP8 as W8A8 natively with no throughput penalty. Halving the weight footprint from ~70 GB to ~35 GB leaves ~87 GB available for KV cache at 0.90 utilisation — roughly 3× more than the BF16 default.

`--kv-cache-dtype fp8_e4m3` is intentionally omitted: Qwen3.5's hybrid MoE+Mamba architecture triggers a vLLM bug at runtime ([vllm-project/vllm#26646](https://github.com/vllm-project/vllm/issues/26646)). Weight-only FP8 already provides the main memory saving.

### Tracing / Prometheus

The `-tracing` variant (`docker-compose.vllm-35b-fp8-rtx-tracing.yml`) adds a Prometheus sidecar alongside the vLLM FP8 service. vLLM exposes metrics at `/metrics` on the same port (11435); Prometheus scrapes it every 15 s and retains data for 15 days.

- **API**: `http://localhost:11435/v1`
- **Metrics endpoint**: `http://localhost:11435/metrics`
- **Prometheus UI**: `http://localhost:9090`

Requires `prometheus.yml` alongside the compose file (included in this directory). Metric data is persisted in a named Docker volume (`qwen35_prometheus_data`). The Prometheus container only starts after vLLM passes its healthcheck.

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

### RTX PRO 6000 Blackwell (96 GB GDDR7)

#### vLLM v0.17.0 FP8 (`docker-compose.vllm-35b-fp8-rtx.yml`)

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

#### vLLM v0.17.0 27B FP8 (`docker-compose.vllm-27b-fp8-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~28 GB FP8 |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~34 tok/s |
| TTFT (with thinking) | ~41 s |

#### vLLM v0.17.0 122B GPTQ-Int4 (`docker-compose.vllm-122b-gptq-int4-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~68 GB GPTQ-Int4 |
| VRAM used | ~89 GB of 96 GB |
| KV cache memory | 11.3 GB |
| KV cache capacity | 121,568 tokens |
| Context length | 131,072 tokens (128K) |
| Decode throughput | ~33 tok/s |
| TTFT (with thinking) | ~49 s |

#### llama.cpp 122B MoE Q4_K_XL (`docker-compose.llama-122b-devfix-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~68 GB Q4_K_XL (3 shards) |
| VRAM used | ~75 GB of 96 GB |
| Context length | 131,072 tokens (128K) |
| KV cache type | q8_0 |
| Decode throughput | ~106 tok/s |
| TTFT (with thinking) | ~12 s |

#### llama.cpp 0.8B dense Q4_K_XL (`docker-compose.llama-0.8b-devfix-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~559 MB Q4_K_XL |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~576 tok/s |
| Notes | Loops in thinking on complex prompts — too small for reasoning tasks |

#### llama.cpp 2B dense Q4_K_XL (`docker-compose.llama-2b-devfix-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~1.3 GB Q4_K_XL |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~381 tok/s |
| TTFT (with thinking) | ~6.0 s |

#### llama.cpp 4B dense Q4_K_XL (`docker-compose.llama-4b-devfix-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~2.9 GB Q4_K_XL |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~228 tok/s |
| TTFT (with thinking) | ~7.0 s |

#### llama.cpp 9B dense Q4_K_XL (`docker-compose.llama-9b-devfix-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~6 GB Q4_K_XL |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~166 tok/s |
| TTFT (with thinking) | ~10.3 s |

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

### RTX 3090 ×2 (48 GB total VRAM)

#### llama.cpp 35B MoE Q4_K_XL (`docker-compose.llama-35b-devfix-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~21 GB Q4_K_XL |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~128 tok/s |
| TTFT (with thinking) | ~10 s |

#### llama.cpp 27B dense Q4_K_XL (`docker-compose.llama-27b-devfix-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~17.6 GB Q4_K_XL |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~39 tok/s |
| TTFT (with thinking) | ~31 s |

#### llama.cpp Qwopus 27B Q4_K_M (`docker-compose.llama-qwopus-27b-rtx.yml`)

| Metric | Value |
|---|---|
| Weights | ~16.5 GB Q4_K_M |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~41 tok/s |
| TTFT (with thinking) | ~15 s |

### AMD Radeon AI PRO R9700 / Vulkan (32 GB GDDR6)

#### llama.cpp 35B MoE Q4_K_XL (`docker-compose.llama-35b-devfix-vulkan.yml`)

| Metric | Value |
|---|---|
| Weights | ~21 GB Q4_K_XL |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~80 tok/s |
| TTFT (with thinking) | ~17 s |

#### llama.cpp 27B dense Q4_K_XL (`docker-compose.llama-27b-devfix-vulkan.yml`)

| Metric | Value |
|---|---|
| Weights | ~17.6 GB Q4_K_XL |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~22 tok/s |
| TTFT (with thinking) | ~56 s |

### Apple Silicon (M-series, 24+ GB unified memory)

#### MLX 35B MoE 4-bit (`run-qwen-mlx.sh`)

| Metric | Value |
|---|---|
| Weights | ~18 GB 4-bit |
| Context length | 262,144 tokens (256K) |
| Decode throughput | ~75 tok/s |
| TTFT (with thinking) | ~350 ms |

> **Note:** MLX is ~2× faster than llama.cpp on Apple Silicon due to native Metal optimization. The extremely low TTFT (~350 ms vs 6–17 s on GPU backends) is because MLX streams tokens immediately without waiting for the full thinking phase to complete.

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
