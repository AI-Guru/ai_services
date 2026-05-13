# Self-Hosted Inference

Runs open-weight models via vLLM, SGLang, llama.cpp, or MLX, exposing an OpenAI-compatible API.

## Models

| Model | Directory | Architecture | Port | Active Params |
|---|---|---|---|---|
| Qwen3.5 (0.8B–122B) | `qwen3.5/` | Dense / MoE | 11430–11437 | 0.8B–10B |
| Qwen3-Coder-Next 80B | `qwen3-coder-next/` | MoE | 11438 | 3B |
| Qwopus 27B | `qwopus/` | Dense | 11436 | 27B |
| GLM-4.7-Flash 30B | `glm-4.7-flash/` | MoE | 11439 | 3.6B |
| Nemotron (4B–120B) | `nemotron/` | Dense / Mamba-2 MoE | 11441–11449 | 3B–30.7B |
| Gemma 4 (E2B–31B) | `gemma4/` | Dense / MoE | 11450–11453 | 2.3B–30.7B |

Each model directory has its own compose files and model-specific README.

## Requirements

- **NVIDIA GPU** — fully tested on DGX Spark GB10 (128 GB unified), RTX PRO 6000 (96 GB GDDR7), and RTX 3090 ×2 (48 GB total); llama.cpp variants work on any CUDA GPU with 24+ GB VRAM
- **AMD GPU** — llama.cpp Vulkan variants tested on AMD Radeon AI PRO R9700 (32 GB GDDR6); requires 24+ GB VRAM
- **Apple Silicon** — MLX variant tested on M-series Macs; requires 24+ GB unified memory for 35B model (32+ GB recommended)
- Docker + NVIDIA Container Toolkit (CUDA) or standard AMD driver stack with `/dev/kfd` + `/dev/dri` (Vulkan)

## Which variant should I use?

```
Which GPU?
├─ DGX Spark (128 GB unified)
│  ├─ Want highest throughput?
│  │  └─ Yes ─────────────────── qwen3.5/llama-35b-devfix-spark.yml (58 tok/s, 23s TTFT)
│  │
│  ├─ Coding agent (Claude Code, Cursor, OpenCode)?
│  │  └─ Yes ─────────────────── qwen3.5/llama-35b-devfix-spark.yml (58 tok/s, patched template)
│  │
│  ├─ Coding agent (Qwen3-Coder-Next)?
│  │  └─ Yes ─────────────────── qwen3-coder-next/llama-coder-next-spark.yml (49 tok/s, coding-specialist, no template patch needed)
│  │
│  ├─ Want largest model (122B)?
│  │  └─ Yes ─────────────────── qwen3.5/vllm-122b-gptq-int4-spark.yml (13 tok/s, 104s TTFT)
│  │
│  ├─ Want dense 27B model?
│  │  ├─ vLLM ────────────────── qwen3.5/vllm-27b-fp8-spark.yml (7.6 tok/s)
│  │  ├─ llama.cpp ───────────── qwen3.5/llama-27b-devfix-spark.yml (11 tok/s, patched template)
│  │  └─ llama.cpp (Qwopus) ─── qwopus/llama-qwopus-27b-spark.yml (11.5 tok/s, Opus distilled)
│  │
│  └─ Otherwise
│     ├─ Fastest ─────────────── qwen3.5/llama-35b-devfix-spark.yml (58 tok/s, 23s TTFT)
│     └─ vLLM ────────────────── qwen3.5/vllm-35b-fp8-spark.yml (48 tok/s, 29s TTFT)
│
├─ RTX PRO 6000 (96 GB GDDR7)
│  ├─ Coding agent (Claude Code, Cursor, OpenCode)?
│  │  └─ Yes ─────────────────── qwen3-coder-next/llama-coder-next-rtx.yml (164 tok/s, coding-specialist, no template patch needed)
│  │                              qwen3.5/llama-35b-devfix-rtx.yml (194 tok/s, general MoE, patched template)
│  │
│  ├─ Want highest throughput?
│  │  └─ Yes ─────────────────── qwen3.5/llama-35b-devfix-rtx.yml (194 tok/s, fastest)
│  │
│  ├─ Want GLM-4.7-Flash (30B MoE, coding/reasoning)?
│  │  ├─ llama.cpp Q4 (fast) ── glm-4.7-flash/llama-q4-rtx.yml (176 tok/s, ~18 GB)
│  │  ├─ llama.cpp Q8 ────────── glm-4.7-flash/llama-q8-rtx.yml (111 tok/s, ~33 GB)
│  │  └─ SGLang FP8 ──────────── glm-4.7-flash/sglang-fp8.yaml (104 tok/s, ~60 GB)
│  │
│  ├─ Want largest model (122B)?
│  │  ├─ llama.cpp (fast) ────── qwen3.5/llama-122b-devfix-rtx.yml (106 tok/s, 12s TTFT)
│  │  └─ vLLM GPTQ ─────────── qwen3.5/vllm-122b-gptq-int4-rtx.yml (33 tok/s, 49s TTFT)
│  │
│  ├─ Want dense 27B model? (65 tok/s, smarter per-token but slower)
│  │  ├─ vLLM ────────────────── qwen3.5/vllm-27b-fp8-rtx.yml (FP8)
│  │  ├─ llama.cpp ───────────── qwen3.5/llama-27b-devfix-rtx.yml (Q4, patched template)
│  │  └─ llama.cpp (Qwopus) ─── qwopus/llama-qwopus-27b-rtx.yml (Opus reasoning distilled)
│  │
│  ├─ Want a small/edge model?
│  │  ├─ 9B dense ───────────── qwen3.5/llama-9b-devfix-rtx.yml (166 tok/s, 10s TTFT)
│  │  ├─ 4B dense ───────────── qwen3.5/llama-4b-devfix-rtx.yml (228 tok/s, 7s TTFT)
│  │  ├─ 2B dense ───────────── qwen3.5/llama-2b-devfix-rtx.yml (381 tok/s, 6s TTFT)
│  │  └─ 0.8B dense ──────────── qwen3.5/llama-0.8b-devfix-rtx.yml (576 tok/s) ⚠ loops on complex prompts
│  │
│  ├─ Want Qwen3-Coder-Next (coding specialist, 80B/3B active)?
│  │  └─ Yes ─────────────────── qwen3-coder-next/llama-coder-next-rtx.yml (164 tok/s, 262K ctx, port 11438)
│  │
│  ├─ Want Prometheus metrics?
│  │  └─ Yes ─────────────────── qwen3.5/vllm-35b-fp8-rtx-tracing.yml (vLLM FP8 + Prometheus on :9090)
│  │
│  └─ Otherwise
│     ├─ Fastest ─────────────── qwen3.5/llama-35b-devfix-rtx.yml (194 tok/s, 6.8s TTFT)
│     └─ vLLM ────────────────── qwen3.5/vllm-35b-fp8-rtx.yml (174 tok/s, 8.5s TTFT)
│
├─ RTX 3090 ×2 (48 GB total VRAM)
│  ├─ Want highest throughput?
│  │  └─ Yes ─────────────────── qwen3.5/llama-35b-devfix-rtx.yml (128 tok/s, 10s TTFT)
│  │
│  ├─ Coding agent (Claude Code, Cursor, OpenCode)?
│  │  └─ Yes ─────────────────── qwen3.5/llama-35b-devfix-rtx.yml (128 tok/s, patched template)
│  │
│  ├─ Want dense 27B model? (smarter per-token but slower)
│  │  ├─ llama.cpp ───────────── qwen3.5/llama-27b-devfix-rtx.yml (39 tok/s, patched template)
│  │  └─ llama.cpp (Qwopus) ─── qwopus/llama-qwopus-27b-rtx.yml (41 tok/s, Opus distilled)
│  │
│  └─ Otherwise
│     └─ Fastest ─────────────── qwen3.5/llama-35b-devfix-rtx.yml (128 tok/s, 10s TTFT)
│
├─ AMD GPU / Vulkan (tested: R9700 32 GB)
│  ├─ Want highest throughput?
│  │  └─ Yes ─────────────────── qwen3.5/llama-35b-devfix-vulkan.yml (80 tok/s, 17s TTFT)
│  │
│  ├─ Coding agent (Claude Code, Cursor, OpenCode)?
│  │  └─ Yes ─────────────────── qwen3.5/llama-35b-devfix-vulkan.yml (80 tok/s, patched template)
│  │
│  ├─ Want Qwen3.6 family (newer arch, hybrid attention)?
│  │  ├─ 35B-A3B (MoE) ───────── qwen3.6/docker-compose.llama-35b-q4-vulkan.yml (109 tok/s, GRAPHICS_QUEUE=1, 32K context)
│  │  ├─ 27B dense ──────────── qwen3.6/docker-compose.llama-27b-q4-vulkan.yml (24.5 tok/s, 64K context)
│  │  ├─ + MTP speculative ──── qwen3.6/docker-compose.llama-{27b,35b}-q4-mtp-vulkan.yml (helps 27B +7 %, hurts 35B −43 %)
│  │  └─ + DFlash (experimental) ─ qwen3.6/docker-compose.dflash-27b-q4-rocm.yml (38 tok/s; ROCm/HIP, gfx1201 untested upstream)
│  │
│  ├─ Want dense 27B model? (smarter per-token but slower)
│  │  └─ llama.cpp ───────────── qwen3.5/llama-27b-devfix-vulkan.yml (22 tok/s, patched template)
│  │
│  └─ Otherwise
│     └─ Fastest ─────────────── qwen3.5/llama-35b-devfix-vulkan.yml (80 tok/s, 17s TTFT)
│
└─ Apple Silicon (tested: M-series 24+ GB)
   └─ run-qwen-mlx.sh ────────── MLX 4-bit (75 tok/s, 350ms TTFT, patched template)
```

> **Note:** llama.cpp variants auto-adjust to available VRAM. On 24 GB GPUs, reduce `-c` (context length) in the compose file to fit — e.g. `-c 8192` instead of `262144`.

### Performance at a glance

#### DGX Spark GB10 (128 GB unified)

| Dir / Compose file | Backend | Model | tok/s | TTFT |
|---|---|---|---|---|
| `qwen3.5/llama-35b-devfix-spark.yml` | llama.cpp Q4_K_XL | Qwen3.5 35B MoE (3B active) | **58.4** | ~23 s |
| `qwen3-coder-next/llama-coder-next-spark.yml` | llama.cpp UD-Q4_K_XL | Qwen3-Coder-Next 80B MoE (3B active) | 49.1 | ~243 ms |
| `qwen3.5/vllm-35b-fp8-spark.yml` | vLLM v0.17.0 FP8 | Qwen3.5 35B MoE (3B active) | 47.9 | ~29 s |
| `qwen3.5/vllm-122b-gptq-int4-spark.yml` | vLLM v0.17.0 GPTQ-Int4 | Qwen3.5 122B MoE (10B active) | 13.0 | ~104 s |
| `qwopus/llama-qwopus-27b-spark.yml` | llama.cpp Q4_K_M | Qwopus 27B (Opus distilled) | 11.5 | ~52 s |
| `qwen3.5/llama-27b-devfix-spark.yml` | llama.cpp Q4_K_XL | Qwen3.5 27B dense | 11.0 | ~130 s |
| `qwen3.5/vllm-27b-fp8-spark.yml` | vLLM v0.17.0 FP8 | Qwen3.5 27B dense | 7.6 | ~173 s |

#### RTX PRO 6000 Blackwell (96 GB GDDR7)

| Dir / Compose file | Backend | Model | tok/s | TTFT |
|---|---|---|---|---|
| `qwen3.5/llama-0.8b-devfix-rtx.yml` | llama.cpp Q4_K_XL | Qwen3.5 0.8B dense | 576.4 | N/A* |
| `qwen3.5/llama-2b-devfix-rtx.yml` | llama.cpp Q4_K_XL | Qwen3.5 2B dense | 380.6 | ~6.0 s |
| `qwen3.5/llama-4b-devfix-rtx.yml` | llama.cpp Q4_K_XL | Qwen3.5 4B dense | 228.4 | ~7.0 s |
| `qwen3.5/llama-35b-devfix-rtx.yml` | llama.cpp Q4_K_XL | Qwen3.5 35B MoE (3B active) | **193.5** | ~6.8 s |
| `glm-4.7-flash/llama-q4-rtx.yml` | llama.cpp Q4_K_XL | GLM-4.7-Flash 30B MoE (3.6B active) | **175.8** | ~6.5 s |
| `qwen3-coder-next/llama-coder-next-rtx.yml` | llama.cpp UD-Q4_K_XL | Qwen3-Coder-Next 80B MoE (3B active) | **164.5** | ~75–610 ms† |
| `qwen3.5/llama-9b-devfix-rtx.yml` | llama.cpp Q4_K_XL | Qwen3.5 9B dense | 165.9 | ~10.3 s |
| `qwen3.5/vllm-35b-fp8-rtx.yml` | vLLM v0.17.0 FP8 | Qwen3.5 35B MoE (3B active) | 156.8 | ~9.6 s |
| `qwen3.5/sglang-fp8.yaml` | SGLang FP8 | Qwen3.5 35B MoE (3B active) | 130.6 | ~9.6 s |
| `glm-4.7-flash/llama-q8-rtx.yml` | llama.cpp Q8_K_XL | GLM-4.7-Flash 30B MoE (3.6B active) | 111.1 | ~11.0 s |
| `qwen3.5/llama-122b-devfix-rtx.yml` | llama.cpp Q4_K_XL | Qwen3.5 122B MoE (10B active) | 105.5 | ~12 s |
| `glm-4.7-flash/sglang-fp8.yaml` | SGLang FP8 | GLM-4.7-Flash 30B MoE (3.6B active) | 104.9 | ~14.8 s |
| `qwopus/llama-qwopus-27b-rtx.yml` | llama.cpp Q4_K_M | Qwopus 27B (Opus distilled) | 68.4 | ~13 s |
| `qwen3.5/llama-27b-devfix-rtx.yml` | llama.cpp Q4_K_XL | Qwen3.5 27B dense | 64.6 | ~21 s |
| `qwen3.5/vllm-27b-fp8-rtx.yml` | vLLM v0.17.0 FP8 | Qwen3.5 27B dense | 34.3 | ~41 s |
| `qwen3.5/vllm-122b-gptq-int4-rtx.yml` | vLLM v0.17.0 GPTQ-Int4 | Qwen3.5 122B MoE (10B active) | 32.6 | ~49 s |
| `gemma4/llama-e2b-rtx.yml` | llama.cpp Q8_0 | Gemma 4 E2B dense (2.3B effective) | 276.3 | ~2.5 s |
| `gemma4/llama-26b-rtx.yml` | llama.cpp UD-Q4_K_XL | Gemma 4 26B MoE (3.8B active) | **196.2** | ~3.8 s |
| `gemma4/llama-e4b-rtx.yml` | llama.cpp Q8_0 | Gemma 4 E4B dense (4.5B effective) | 171.0 | ~1.8 s |
| `gemma4/llama-31b-rtx.yml` | llama.cpp Q4_K_M | Gemma 4 31B dense (30.7B) | 64.6 | ~9.0 s |
| `gemma4/vllm-31b-nvfp4-rtx.yml` | vLLM NVFP4 | Gemma 4 31B dense (30.7B) | 39.3 | ~95 ms |

\* 0.8B loops in thinking on complex prompts — too small for reasoning tasks.
† Qwen3-Coder-Next TTFT is 75–150 ms warm (KV cache hot) and ~610 ms cold. No thinking preamble observed at default settings.

#### RTX 3090 ×2 (48 GB total VRAM)

| Dir / Compose file | Backend | Model | tok/s | TTFT |
|---|---|---|---|---|
| `qwen3.5/llama-35b-devfix-rtx.yml` | llama.cpp Q4_K_XL | Qwen3.5 35B MoE (3B active) | **127.6** | ~10 s |
| `qwopus/llama-qwopus-27b-rtx.yml` | llama.cpp Q4_K_M | Qwopus 27B (Opus distilled) | 41.0 | ~15 s |
| `qwen3.5/llama-27b-devfix-rtx.yml` | llama.cpp Q4_K_XL | Qwen3.5 27B dense | 39.2 | ~31 s |

#### AMD Radeon AI PRO R9700 / Vulkan (32 GB GDDR6)

| Dir / Compose file | Backend | Model | tok/s | TTFT |
|---|---|---|---|---|
| `qwen3.5/llama-35b-devfix-vulkan.yml` | llama.cpp Q4_K_XL Vulkan | Qwen3.5 35B MoE (3B active) | **79.9** | ~17 s |
| `qwen3.5/llama-27b-devfix-vulkan.yml` | llama.cpp Q4_K_XL Vulkan | Qwen3.5 27B dense | 21.9 | ~56 s |
| `qwen3.6/docker-compose.llama-27b-q4-vulkan.yml` | llama.cpp UD-Q4_K_XL Vulkan (2026-05 image) | Qwen3.6 27B dense | 24.5 | — |
| `qwen3.6/docker-compose.llama-27b-q4-mtp-vulkan.yml` | llama.cpp UD-Q4_K_XL Vulkan + MTP | Qwen3.6 27B dense | 26.1 (+7 %) | — |
| `qwen3.6/docker-compose.llama-35b-q4-vulkan.yml` | llama.cpp UD-Q4_K_XL Vulkan + `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | Qwen3.6 35B-A3B MoE (3B active) | **109.3** | — |
| `qwen3.6/docker-compose.llama-35b-q4-mtp-vulkan.yml` | llama.cpp UD-Q4_K_XL Vulkan + MTP | Qwen3.6 35B-A3B MoE (3B active) | 62.2 (−43 %) | — |
| `qwen3.6/docker-compose.dflash-27b-q4-rocm.yml` | Lucebox DFlash HIP (gfx1201, experimental) | Qwen3.6 27B dense | **38.0** | ~0.5 s |

#### Apple Silicon (M-series, 24+ GB unified memory)

| Script | Backend | Model | tok/s | TTFT |
|---|---|---|---|---|
| `qwen3.5/run-qwen-mlx.sh` | mlx-openai-server 4-bit | Qwen3.5 35B MoE (3B active) | **74.7** | ~350 ms |

Measured with `shared/test_chat.py --warmup --runs 3`.

## Port allocation

Ports are assigned per model — you can run multiple models side by side:

| Model | Port |
|---|---|
| Qwen3.5 0.8B | 11430 |
| Qwen3.5 2B | 11431 |
| Qwen3.5 4B | 11432 |
| Qwen3.5 9B | 11433 |
| Ollama | 11434 |
| Qwen3.5 35B | 11435 |
| Qwen3.5 27B / Qwopus | 11436 |
| Qwen3.5 122B | 11437 |
| Qwen3-Coder-Next | 11438 |
| GLM-4.7-Flash | 11439 |

## Quick start

### Docker (NVIDIA / AMD)

```bash
# Qwen3.5 35B — vLLM BF16, 262K context, multimodal (port 11435)
cd qwen3.5 && docker compose -f docker-compose.vllm.yaml up -d

# Qwen3.5 35B — SGLang FP8 (port 11435)
cd qwen3.5 && docker compose -f docker-compose.sglang-fp8.yaml up -d

# GLM-4.7-Flash — llama.cpp Q4 (port 11439)
cd glm-4.7-flash && docker compose -f docker-compose.llama-q4-rtx.yml up -d

# GLM-4.7-Flash — SGLang FP8 (port 11439)
cd glm-4.7-flash && docker compose -f docker-compose.sglang-fp8.yaml up -d

# Watch startup (model load takes 3–5 min on first run)
docker compose logs -f
```

API is ready when logs show `Application startup complete` (vLLM/SGLang) or `all slots are idle` (llama.cpp).

### Apple Silicon (MLX)

MLX requires native Metal GPU access — no Docker. The `run-qwen-mlx.sh` script manages a conda environment.

```bash
cd qwen3.5

# One-time setup (creates conda env, installs mlx-openai-server)
./run-qwen-mlx.sh --install

# Start server (downloads ~18 GB model on first run)
./run-qwen-mlx.sh --serve

# Other commands
./run-qwen-mlx.sh --upgrade    # Upgrade MLX packages
./run-qwen-mlx.sh --uninstall  # Remove conda env
```

Requires: conda (Miniconda or Anaconda), 24+ GB unified memory (32+ GB recommended).

## Qwen3.5 variants

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
| llama.cpp Qwen3-Coder-Next | `docker compose -f docker-compose.llama-coder-next-rtx.yml up -d` | `qwen3-coder-next` | ~50 GB UD-Q4_K_XL | 80B/3B-active MoE; coding specialist; 262K ctx; port 11438 |
| llama.cpp Qwen3-Coder-Next Spark | `docker compose -f docker-compose.llama-coder-next-spark.yml up -d` | `qwen3-coder-next` | ~50 GB UD-Q4_K_XL | DGX Spark ARM64; coding specialist; 262K ctx; port 11438 |
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

All Qwen variants share the same named Docker volume (`qwen35_huggingface_cache`) so weights are only downloaded once per model variant.

## GLM-4.7-Flash variants

| Variant | Command | Weights | Notes |
|---|---|---|---|
| llama.cpp Q4 | `docker compose -f docker-compose.llama-q4-rtx.yml up -d` | ~18 GB Q4_K_XL | Fastest; recommended for single-user |
| llama.cpp Q8 | `docker compose -f docker-compose.llama-q8-rtx.yml up -d` | ~33 GB Q8_K_XL | Higher quality; more VRAM |
| vLLM FP8 | `docker compose -f docker-compose.vllm-fp8.yml up -d` | ~30 GB FP8 | Continuous batching; MTP speculative decoding |
| vLLM BF16 | `docker compose -f docker-compose.vllm-bf16.yml up -d` | ~70 GB BF16 | Full precision; most VRAM |
| SGLang FP8 | `docker compose -f docker-compose.sglang-fp8.yaml up -d` | ~60 GB BF16 + FP8 KV | Continuous batching; triton attention |

All GLM variants share the named Docker volume (`glm47_huggingface_cache`). Create it before first use: `docker volume create glm47_huggingface_cache`.

vLLM and SGLang GLM variants require a custom Docker image (built automatically on first `docker compose up`) because the `glm4_moe_lite` architecture is not yet in a released `transformers` version.

## Qwen3.5 notes

### llama.cpp

> Based on findings by [@sudoingX](https://x.com/sudoingx) regarding the Qwen 3.5 jinja template crash with coding agents.

The stock Qwen 3.5 chat template **crashes on the `developer` role** that coding agents (Claude Code, OpenCode, Cursor) send. The common workaround `--chat-template chatml` silently kills thinking mode. The `llama-35b` variant uses a [patched jinja template](qwen3.5/qwen3.5_chat_template.jinja) that adds `developer` role handling while preserving `<think>` blocks. The `llama-qwopus-27b` variant is a [community fine-tune](https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF) with Claude Opus 4.6 reasoning distilled in that handles the developer role natively (experimental, no formal benchmarks).

All llama.cpp variants auto-download their GGUF model on first start. The `-rtx` variants use the official x86_64 image (`ghcr.io/ggml-org/llama.cpp:server-cuda`); the `-spark` variants use a community ARM64 CUDA image (`ghcr.io/ardge-labs/llama-cpp-dgx-spark:server`) for DGX Spark's aarch64 architecture; the `-vulkan` variants use the official Vulkan image (`ghcr.io/ggml-org/llama.cpp:server-vulkan`) which works on AMD GPUs without ROCm.

### Vulkan

The `-vulkan` variants use `ghcr.io/ggml-org/llama.cpp:server-vulkan` and rely on the standard AMD GPU driver stack — no CUDA or ROCm installation required. GPU access is provided by passing `/dev/kfd` and `/dev/dri` into the container with the host video (GID 44) and render (GID 992) groups.

> **Note:** `radv is not a conformant Vulkan implementation` is printed at startup by the RADV Mesa driver — this is a warning only, not an error. The model runs correctly.

Vulkan throughput for the 35B MoE model on the R9700 (79.9 tok/s) exceeds the DGX Spark (58.4 tok/s) but trails the dual RTX 3090 (127.6 tok/s). The 27B dense model is significantly slower on Vulkan (21.9 tok/s) than on CUDA due to the higher memory-bandwidth demands of a dense model and less mature Vulkan kernel optimizations for that workload.

If the container fails to start, try removing `-fa on` (flash attention) — support varies across Vulkan implementations.

### FP8

The `fp8` profile uses the official Qwen pre-quantized model ([Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) rather than on-the-fly quantization. GB10 (Blackwell, CC 12.1) runs block-wise FP8 as W8A8 natively with no throughput penalty. Halving the weight footprint from ~70 GB to ~35 GB leaves ~87 GB available for KV cache at 0.90 utilisation — roughly 3× more than the BF16 default.

`--kv-cache-dtype fp8_e4m3` is intentionally omitted: Qwen3.5's hybrid MoE+Mamba architecture triggers a vLLM bug at runtime ([vllm-project/vllm#26646](https://github.com/vllm-project/vllm/issues/26646)). Weight-only FP8 already provides the main memory saving.

### Tracing / Prometheus

The `-tracing` variant (`docker-compose.vllm-35b-fp8-rtx-tracing.yml`) adds a Prometheus sidecar alongside the vLLM FP8 service. vLLM exposes metrics at `/metrics` on the same port (11435); Prometheus scrapes it every 15 s and retains data for 15 days.

- **API**: `http://localhost:11435/v1`
- **Metrics endpoint**: `http://localhost:11435/metrics`
- **Prometheus UI**: `http://localhost:9090`

Requires `prometheus.yml` alongside the compose file (included in the `qwen3.5/` directory). Metric data is persisted in a named Docker volume (`qwen35_prometheus_data`). The Prometheus container only starts after vLLM passes its healthcheck.

### MXFP4

No vLLM-compatible MXFP4 checkpoint exists for this model yet. A GGUF variant is available ([noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF](https://huggingface.co/noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF), ~22 GB) but requires llama.cpp. Additionally, vLLM's native NVFP4 kernel support on GB10 (SM12.x) is still maturing ([vllm-project/vllm#31085](https://github.com/vllm-project/vllm/issues/31085)).

## API

All models expose an OpenAI-compatible API. Port depends on the model (see port table above).

```bash
# Qwen3.5 35B (port 11435)
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# GLM-4.7-Flash (port 11439)
curl http://localhost:11439/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7-flash",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Other useful endpoints:

```bash
curl http://localhost:11435/health          # liveness
curl http://localhost:11439/v1/models       # confirm model name
```

## Test scripts

```bash
pip install openai

# Chat latency — TTFT and tokens/sec
python shared/test_chat.py
python shared/test_chat.py --runs 5
python shared/test_chat.py --base-url http://localhost:11439/v1 --model glm-4.7-flash

# Tool calling — single, parallel, chained, and multi-parallel scenarios
python shared/test_tools.py
python shared/test_tools.py --scenario single
python shared/test_tools.py --base-url http://localhost:11439/v1 --model glm-4.7-flash
```

## Environment variables

Create a `.env` file alongside the compose files to override defaults:

```dotenv
HF_TOKEN=hf_...              # required for gated models; not needed for Qwen3.5 or GLM-4.7
HF_HUB_OFFLINE=1             # set after first successful download to block runtime fetches
VLLM_LOGGING_LEVEL=WARNING   # reduce log noise in production
```

## Troubleshooting

**Container stays `unhealthy`**
Large models take longer than the healthcheck `start_period` on first run while weights are being downloaded. Check actual progress with `docker compose logs -f` and wait for startup to complete. The container will self-heal once the API is up.

**High swap usage**
Normal during model load on DGX Spark. Swap should stabilise once weights are fully resident in unified memory. If swap keeps climbing, lower `--gpu-memory-utilization` (e.g. from `0.80` to `0.75`).

**`fp8e4nv not supported` error**
This is the KV cache FP8 bug ([#26646](https://github.com/vllm-project/vllm/issues/26646)). The Qwen3.5 FP8 profile already omits `--kv-cache-dtype`; if you have added it manually, remove it.

**PyTorch CUDA capability warning (GB10 / CC 12.1)**
`Found GPU0 NVIDIA GB10 which is of cuda capability 12.1. Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)` — this is a warning only, not an error. The model runs correctly.

**GLM-4.7-Flash `glm4_moe_lite` not recognized**
vLLM and SGLang require `transformers` installed from git for GLM-4.7-Flash support. The compose files handle this automatically via inline Dockerfiles. If building manually, run: `pip install git+https://github.com/huggingface/transformers.git`.
