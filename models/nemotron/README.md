# Nemotron Self-Hosted Inference

Runs NVIDIA Nemotron models via vLLM or llama.cpp on NVIDIA Blackwell GPUs, exposing an OpenAI-compatible API.

## Models

### Nemotron-Cascade-2-30B-A3B (Mar 18, 2026)

The standout release. Same form factor as Qwen3.5-35B-A3B (30B MoE, ~3B active) but dramatically stronger on coding and math. Hybrid Mamba-2 + MoE + Attention architecture (NemotronH). Post-trained from Nemotron-3-Nano-30B base with thinking mode.

### Nemotron-3-Nano-30B-A3B (Dec 2025)

The base model behind Cascade-2. General-purpose chat, RAG, and tool use. Well-tested with 1.4M+ FP8 downloads. Supports 1M context. Official FP8 and NVFP4 quantizations from NVIDIA.

### Nemotron-Terminal-8B / 14B / 32B (Feb 2026)

Qwen3 fine-tuned for terminal/CLI agent tasks. Dense Transformer (no Mamba, no MoE). Standard vLLM compatibility with no special flags.

### Nemotron-Cascade-8B (Dec 2025)

Dense 8B model (Qwen3-8B based) optimized for reasoning. Best-in-class 8B on LiveCodeBench (71.1) and AIME25 (80.1).

### Nemotron-3-Nano-4B (Mar 2026)

Edge/IoT model. Dense hybrid Mamba-2 + Attention (42 layers, only 4 attention). Distilled from Nemotron-Nano-9B-v2 via Elastic framework. Official GGUF available.

### Nemotron-3-Super-120B-A12B (Mar 2026)

Most capable overall. LatentMoE with 120B/12B active + MTP speculative decoding. Does NOT fit on a single RTX PRO 6000 (needs 8x H100-80GB minimum). Included for reference.

## Requirements

- NVIDIA GPU with 24+ GB VRAM (RTX PRO 6000 96 GB recommended for full context)
- Docker + NVIDIA Container Toolkit
- `vllm/vllm-openai:v0.17.0-cu130` or `ghcr.io/ggml-org/llama.cpp:server-cuda` or `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc8`

## Which variant should I use?

```
What do you need?
├─ Best coding + math (single/pair dev)?
│  ├─ llama.cpp (fastest) ──── llama-cascade2-30b-rtx.yml (280 tok/s)
│  └─ vLLM (multi-user) ───── vllm-cascade2-30b-awq-int4-rtx.yml (351 tok/s, scales to 10+ users)
│
├─ General chat/RAG (team, high concurrency)?
│  ├─ llama.cpp ────────────── llama-nano-30b-rtx.yml (248 tok/s)
│  ├─ vLLM FP8 ─────────────── vllm-nano-30b-fp8-rtx.yml
│  └─ vLLM NVFP4 ───────────── vllm-nano-30b-nvfp4-rtx.yml (max KV cache)
│
├─ Terminal/CLI agent?
│  ├─ llama.cpp 32B ─────────── llama-terminal-32b-rtx.yml (65 tok/s)
│  ├─ vLLM 32B ──────────────── vllm-terminal-32b-rtx.yml (strongest)
│  ├─ vLLM 14B ──────────────── vllm-terminal-14b-rtx.yml (balanced)
│  └─ vLLM 8B ───────────────── vllm-terminal-8b-rtx.yml (fastest)
│
├─ Reasoning (small, fast)?
│  ├─ llama.cpp ────────────── llama-cascade-8b-rtx.yml (147 tok/s)
│  └─ vLLM ──────────────────── vllm-cascade-8b-rtx.yml
│
├─ Reasoning (14B, thinking-only)?
│  ├─ llama.cpp ────────────── llama-cascade-14b-rtx.yml (89 tok/s)
│  └─ vLLM ──────────────────── vllm-cascade-14b-rtx.yml
│
├─ Edge / IoT?
│  ├─ llama.cpp ────────────── llama-nano-4b-rtx.yml (299 tok/s)
│  └─ vLLM ──────────────────── vllm-nano-4b-rtx.yml
│
└─ Maximum capability (single GPU, tight fit)?
   └─ Super-120B NVFP4 ────── vllm-super-120b-nvfp4-rtx.yml (80 tok/s, 95/98 GB VRAM)
```

## Performance at a glance

### RTX PRO 6000 Blackwell (96 GB GDDR7)

#### llama.cpp (single-user, `test_scenarios.py --no-think`, tok/s by scenario)

| Compose file | Model | Quant | Chat | RAG | Codegen | Summary | Agentic | **Avg** |
|---|---|---|---|---|---|---|---|---|
| `llama-nano-4b-rtx.yml` | Nano 4B dense | Q4_K_M | 230 | 312 | 322 | 309 | 319 | **299** |
| `llama-cascade2-30b-rtx.yml` | Cascade-2 30B MoE | Q4_K_M | 163 | 269 | 281 | 263 | 277 | **250** |
| `llama-nano-30b-rtx.yml` | Nano 30B MoE | Q4_K_XL | 153 | 269 | 280 | 265 | 276 | **249** |
| `llama-cascade-8b-rtx.yml` | Cascade 8B dense | Q8_0 | 124 | 146 | 152 | 157 | 151 | **146** |
| `llama-cascade-14b-rtx.yml` | Cascade 14B dense | Q8_0 | 83 | 90 | 91 | 93 | 90 | **89** |
| `llama-terminal-32b-rtx.yml` | Terminal 32B dense | Q4_K_M | 62 | 64 | 65 | 66 | 64 | **64** |

Scenarios:
- **Chat** — short Q&A (~55 tok in, ~300 tok out)
- **RAG** — retrieval with 4 doc chunks (~775 tok in, ~300 tok out)
- **Codegen** — file context to full function (~150 tok in, ~1000 tok out)
- **Summary** — long document to bullet points (~577 tok in, ~200 tok out)
- **Agentic** — tool-use agent with history + results (~895 tok in, ~600 tok out)

MoE models (Cascade-2, Nano 30B) achieve 2-4x the throughput of equivalently-sized dense models because only 3B params are active per token. Nano 4B is fastest overall but quality is limited by model size.

Measured with `test_scenarios.py --runs 3 --warmup --no-think` (default).

#### vLLM (single-user, `test_scenarios.py --no-think`, tok/s by scenario)

| Compose file | Model | Quant | Chat | RAG | Codegen | Summary | Agentic | **Avg** |
|---|---|---|---|---|---|---|---|---|
| `vllm-cascade2-30b-awq-int4-rtx.yml` | Cascade-2 30B MoE | AWQ-INT4 | 269 | 273 | 277 | 268 | 278 | **273** |
| `vllm-nano-30b-fp8-rtx.yml` | Nano 30B MoE | FP8 | 255 | 249 | 271 | 212 | 265 | **250** |
| `vllm-nano-30b-nvfp4-rtx.yml` | Nano 30B MoE | NVFP4 | 260 | 248 | 264 | 210 | 260 | **248** |
| `vllm-super-120b-nvfp4-rtx.yml` | Super 120B MoE (12B active) | NVFP4 | 82 | 80 | 83 | 74 | 82 | **80** |

Nano-30B FP8 and NVFP4 deliver identical throughput (~250 tok/s). NVFP4 uses less VRAM (19 GB vs 33 GB), leaving ~67 GB for KV cache — best for high concurrency.

Super-120B NVFP4 fits on a single RTX PRO 6000 (95/98 GB VRAM) with ~2 GB to spare. No KV cache room for concurrency — single-user only. TTFT is excellent (56-164ms) thanks to NVFP4 prefill efficiency.

#### TensorRT-LLM (single-user, `test_scenarios.py`, tok/s by scenario)

| Compose file | Model | Backend | Chat | RAG | Codegen | Summary | Agentic | **Avg** |
|---|---|---|---|---|---|---|---|---|
| `trtllm-cascade-8b-rtx.yml` | Cascade 8B dense | PyTorch BF16 | 91 | 91 | 92 | 91 | 91 | **91** |
| `trtllm-terminal-8b-rtx.yml` | Terminal 8B dense | PyTorch BF16 | 91 | 91 | 91 | 91 | 91 | **91** |
| `trtllm-cascade-14b-rtx.yml` | Cascade 14B dense | PyTorch BF16 | 52 | 51 | 52 | 52 | 52 | **52** |
| `trtllm-terminal-14b-rtx.yml` | Terminal 14B dense | PyTorch BF16 | 52 | 51 | 52 | 51 | 51 | **51** |
| `trtllm-terminal-32b-rtx.yml` | Terminal 32B dense | PyTorch BF16 | 23 | 23 | 23 | 23 | 23 | **23** |

TensorRT-LLM 1.3.0rc8 PyTorch backend is significantly slower than both llama.cpp and vLLM on all dense models. The TensorRT compiled backend (`--backend tensorrt`) builds optimized engines but has issues on SM120: Cascade-14B OOMs during engine compilation, and NemotronH models are not supported (`NemotronHForCausalLM` not implemented). Compiled backend testing is still in progress for 8B models.

#### vLLM (multi-user throughput — Cascade-2-30B AWQ-INT4)

| Scenario | Input | Output | Single-user tok/s | Single-user latency | Sweet spot conc. | Practical users |
|---|---|---|---|---|---|---|
| Chat | 2K | 300 | **351** | 0.9s | ~21 | **10-15** |
| RAG | 8K | 256 | **277** | 0.9s | ~1 | **1-2** |
| Codegen | 4K | 1500 | **373** | 4.0s | ~2 | **1-2** |
| Agentic | 16K | 800 | **302** | 2.5s | ~1 | **1** |
| Summarization | 12K | 300 | **254** | 1.2s | ~1 | **1** |

Measured with `test_chat.py --warmup --runs 3` (llama.cpp) and `guidellm benchmark --profile sweep` (vLLM).

### Comparison with Qwen3.5 FP8

| Metric | Qwen3.5 FP8 (vLLM) | Cascade-2 AWQ (vLLM) | Cascade-2 (llama.cpp) |
|---|---|---|---|
| TPOT (chat) | 6.0ms | **2.8ms** | ~3.6ms |
| Single-user tok/s (chat) | 146 | **351** | **280** |
| Chat concurrency | ~15 | **~21** | 1 (single slot) |
| Chat users | 5-8 | **10-15** | **1** |
| KV cache | 51 GB | 67 GB | — |
| Model weights | 35 GB FP8 | ~17 GB AWQ | ~25 GB Q4 |

## Upstream benchmarks

### Coding and reasoning

| Benchmark | Cascade-2-30B | Nano-30B | Qwen3.5-35B | Qwen3-Coder-Next |
|---|---|---|---|---|
| LiveCodeBench v6 | **87.2** | 68.3 | 74.6 | — |
| AIME 2025 | **92.4** | 89.1 | 91.9 | — |
| IOI 2025 | **439 (Gold)** | — | 349 | — |
| ArenaHard v2 | **83.5** | 67.7 | 65.4 | — |
| IFBench | **82.9** | 71.5 | 70.2 | — |
| SWE-Bench Verified | 50.2 | 38.8 | **69.2** | **70.6** |

### General knowledge

| Benchmark | Cascade-2-30B | Nano-30B | Qwen3.5-35B |
|---|---|---|---|
| MMLU-Pro | 79.8 | 78.3 | **85.3** |
| NIAH @1M | **99.0** | 94.8 | 94.3 |
| BFCL v4 (tool use) | 52.9 | **53.8** | **67.3** |
| tau2-Bench (agentic) | 58.9 | 49.0 | **81.2** |

## VRAM fit on RTX PRO 6000 (96 GB)

| Model | Quantization | Weights | KV cache room | Fits? |
|---|---|---|---|---|
| Cascade-2-30B | AWQ-INT4 (community) | ~17 GB | ~69 GB | Yes — best headroom |
| Cascade-2-30B | Q4_K_M GGUF | ~25 GB | — | Yes |
| Cascade-2-30B | FP8 (community) | ~32 GB | ~54 GB | Yes |
| Cascade-2-30B | BF16 | ~63 GB | ~23 GB | Yes — tight |
| Nano-30B | Q4_K_XL GGUF | ~23 GB | — | Yes |
| Nano-30B | NVFP4 (official) | 19.3 GB | ~67 GB | Yes |
| Nano-30B | FP8 (official) | 32.7 GB | ~53 GB | Yes |
| Nano-30B | BF16 | ~63 GB | ~23 GB | Yes — tight |
| Cascade-8B | Q8_0 GGUF | ~9 GB | — | Yes |
| Cascade-8B | BF16 | ~16 GB | ~70 GB | Yes |
| Terminal-8B | BF16 | ~16 GB | ~70 GB | Yes |
| Terminal-14B | BF16 | ~28 GB | ~58 GB | Yes |
| Terminal-32B | Q4_K_M GGUF | ~20 GB | — | Yes |
| Terminal-32B | BF16 | ~64 GB | ~22 GB | Yes — tight |
| Nano-4B | BF16 | ~8 GB | ~78 GB | Yes |
| Super-120B | NVFP4 | ~62 GB | ~24 GB | Tight — minimal KV |
| Super-120B | FP8 | ~124 GB | — | No |

## Quick start

```bash
# Cascade-2-30B llama.cpp — fastest single-user (port 11440)
docker compose -f docker-compose.llama-cascade2-30b-rtx.yml up -d

# Cascade-2-30B vLLM AWQ — best for teams (port 11440)
docker compose -f docker-compose.vllm-cascade2-30b-awq-int4-rtx.yml up -d

# Watch startup
docker compose logs -f
```

API is ready when healthcheck passes or logs show `Application startup complete`.

## Service variants

All Nemotron models use port 11440 to avoid conflicts with Qwen3.5 (11435) and Qwen3-Coder-Next (11438).

| Variant | Command | Model | Weights |
|---|---|---|---|
| Cascade-2-30B llama.cpp | `docker compose -f docker-compose.llama-cascade2-30b-rtx.yml up -d` | `nemotron-cascade2-30b` | ~25 GB Q4_K_M |
| Cascade-2-30B vLLM AWQ | `docker compose -f docker-compose.vllm-cascade2-30b-awq-int4-rtx.yml up -d` | `nemotron-cascade2-30b` | ~17 GB AWQ |
| Nano-30B llama.cpp | `docker compose -f docker-compose.llama-nano-30b-rtx.yml up -d` | `nemotron-nano-30b` | ~23 GB Q4_K_XL |
| Nano-30B vLLM FP8 | `docker compose -f docker-compose.vllm-nano-30b-fp8-rtx.yml up -d` | `nemotron-nano-30b` | ~33 GB FP8 |
| Nano-30B vLLM NVFP4 | `docker compose -f docker-compose.vllm-nano-30b-nvfp4-rtx.yml up -d` | `nemotron-nano-30b` | ~19 GB NVFP4 |
| Cascade-8B llama.cpp | `docker compose -f docker-compose.llama-cascade-8b-rtx.yml up -d` | `nemotron-cascade-8b` | ~9 GB Q8_0 |
| Cascade-8B vLLM | `docker compose -f docker-compose.vllm-cascade-8b-rtx.yml up -d` | `nemotron-cascade-8b` | ~16 GB BF16 |
| Terminal-32B llama.cpp | `docker compose -f docker-compose.llama-terminal-32b-rtx.yml up -d` | `nemotron-terminal-32b` | ~20 GB Q4_K_M |
| Terminal-32B vLLM | `docker compose -f docker-compose.vllm-terminal-32b-rtx.yml up -d` | `nemotron-terminal-32b` | ~64 GB BF16 |
| Terminal-14B vLLM | `docker compose -f docker-compose.vllm-terminal-14b-rtx.yml up -d` | `nemotron-terminal-14b` | ~28 GB BF16 |
| Terminal-8B vLLM | `docker compose -f docker-compose.vllm-terminal-8b-rtx.yml up -d` | `nemotron-terminal-8b` | ~16 GB BF16 |
| Nano-4B vLLM | `docker compose -f docker-compose.vllm-nano-4b-rtx.yml up -d` | `nemotron-nano-4b` | ~8 GB BF16 |

All variants share the `qwen35_huggingface_cache` Docker volume so weights are only downloaded once per model variant.

## llama.cpp notes

### NemotronH architecture (Cascade-2-30B, Nano-30B, Nano-4B)

NemotronH (hybrid Mamba-2 + MoE + Attention) is supported in llama.cpp since b6315+ (PR #18058, Dec 2025). The Mamba-2 assertion bug (#20570) was fixed on Mar 15, 2026 — requires the `ghcr.io/ggml-org/llama.cpp:server-cuda` image built after that date.

- GPU (CUDA) is required — CPU-only has unresolved issues
- Use `--special` flag to see `<think>`/`</think>` reasoning tokens
- `--jinja` is NOT required
- Context can go up to 262K but large contexts increase VRAM usage significantly

### Dense Transformer models (Terminal-32B, Cascade-8B)

Standard Qwen3 architecture. No special flags needed. These just work with any recent llama.cpp build.

## vLLM notes

### NemotronH architecture (Nano-30B, Cascade-2-30B, Nano-4B)

vLLM requirements:
- `--trust-remote-code` (custom model code)
- `--attention-backend FLASHINFER` (required for NemotronH)
- Prefix caching is experimental for Mamba layers — enabled with `mamba_ssm_cache_dtype` alignment
- CUDA graphs work (no `--enforce-eager` needed)

### Dense Transformer models (Terminal, Cascade-8B)

Standard Qwen3 architecture. No special flags needed beyond the usual vLLM config.

## TensorRT-LLM notes

Image: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc8` (21 GB). Must use the default NVIDIA entrypoint (do not override with `entrypoint:`) — it sets up `LD_LIBRARY_PATH` for TensorRT libraries.

### Supported models (dense Qwen3 only)

Cascade-8B, Cascade-14B, Terminal-8B/14B/32B. These work with both `--backend pytorch` (default) and `--backend tensorrt` (compiled engines).

### Not supported: NemotronH architecture

Cascade-2-30B, Nano-30B, Nano-4B, Super-120B all use `NemotronHForCausalLM` which is not implemented in TRT-LLM 1.3.0rc8. The PyTorch backend can load them but performance is poor. The TensorRT compiled backend fails with `NotImplementedError`.

### SM120 (RTX Pro 6000) limitations

- NVFP4 MoE kernels: SM100 only (datacenter Blackwell)
- NVFP4 KV cache: SM100 only
- Engine build for 14B+ models may OOM during compilation (compiler + weights exceed GPU memory)

## API

```bash
curl http://localhost:11440/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-cascade2-30b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```
