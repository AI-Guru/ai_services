# Qwen3.6 Family

Two open-weight variants from Alibaba Qwen (Apache 2.0):

- **[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** — 35B-total / 3B-active MoE (fast, cheap to serve)
- **[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)** — 27B dense (stronger per-token, bandwidth-bound)

Both share the hybrid Gated DeltaNet + Gated Attention architecture and the same reasoning / tool-calling parsers (`qwen3` / `qwen3_coder`).

---

# Qwen3.6-35B-A3B

[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — 35B-total / 3B-active Mixture-of-Experts model from Alibaba Qwen.

## Architecture

- **Hybrid attention**: 30 Gated DeltaNet (linear attention) + 10 full Gated Attention layers in a repeating 3:1 pattern
- **MoE**: 256 experts, 8 routed + 1 shared active per token
- **Multimodal**: text, image, and video input (up to 224K video tokens)
- **Context**: 262K native, extensible to ~1M via YaRN
- **Multi-Token Prediction**: native MTP for speculative decoding
- **License**: Apache 2.0

## Quick start

```bash
# RTX PRO 6000 (96 GB) — primary config
docker compose -f docker-compose.vllm-35b-fp8-rtx.yml up -d

# Apple Silicon (24+ GB unified memory)
./run-qwen-mlx.sh --install
./run-qwen-mlx.sh --serve
```

API endpoint: `http://localhost:11435/v1` (vLLM/llama.cpp/SGLang), `http://localhost:11436/v1` (MLX), model name: `qwen3.6-35b`

## MLX (Apple Silicon)

Script: `run-qwen-mlx.sh` — uses `mlx-openai-server` with `qwen3.6_chat_template.jinja`, port **11436**.

| Model | Size | Memory |
|-------|-----:|-------:|
| `mlx-community/Qwen3.6-35B-A3B-4bit` *(default)* | ~20 GB | 24+ GB |
| `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ` | ~20 GB | 24+ GB |
| `mlx-community/Qwen3.6-35B-A3B-6bit` | ~30 GB | 40+ GB |
| `mlx-community/Qwen3.6-35B-A3B-8bit` | ~40 GB | 48+ GB |
| `mlx-community/Qwen3.6-35B-A3B-bf16` | ~70 GB | 96+ GB |

### MLX Benchmarks (Apple Silicon, 2026-04-20)

Tested with `test_chat.py` (3 runs + warmup), `mlx-openai-server` 1.7.1, `mlx-lm` 0.31.2. tok/s counts all generated tokens (think + answer) over full wall time.

| Run | TTFT | tok/s | Think time |
|-----|-----:|------:|-----------:|
| 1 | 25.4s | 80.1 | 25.2s |
| 2 | 19.1s | 79.9 | 18.9s |
| 3 | 22.6s | 79.7 | 22.4s |
| **avg** | **22.4s** | **79.9** | **22.2s** |

Think time dominates TTFT — the model reasons extensively before answering. Decode throughput is consistent at ~80 tok/s including think tokens.

## Compose variants

| File | Engine | Quant | Hardware | Notes |
|------|--------|-------|----------|-------|
| `docker-compose.vllm-35b-fp8-rtx.yml` | vLLM | FP8 | RTX PRO 6000 | Primary config, 262K context |
| `docker-compose.vllm-35b-fp8-textonly-rtx.yml` | vLLM | FP8 | RTX PRO 6000 | FP8 + vision disabled, fastest vLLM variant |
| `docker-compose.vllm-35b-nvfp4-unsloth-rtx.yml` | vLLM | NVFP4 | RTX PRO 6000 | Unsloth NVFP4 (group_size=8); pre-fused experts, pure NVFP4 |
| `docker-compose.vllm-35b-nvfp4-nvidia-rtx.yml` | vLLM | NVFP4 (modelopt_mixed) | RTX PRO 6000 | Official NVIDIA NVFP4 without MTP, +40% over Unsloth. Entrypoint pip-upgrades vLLM nightly + flashinfer 0.6.12 (jit-cache from `flashinfer.ai/whl/nightly/cu130`) on container start; ~60s overhead, no custom image |
| `docker-compose.vllm-35b-nvfp4-nvidia-mtp-rtx.yml` | vLLM | NVFP4 + MTP | RTX PRO 6000 | **Recommended default for 35B Qwen3.6 — 367.7 tok/s, +55% over no-MTP, +116% over Unsloth.** Same loader recipe + `--speculative-config method=mtp num=3 triton-drafter`. Port 11438 |
| `docker-compose.vllm-35b-text-only.yaml` | vLLM | BF16 | Any | Vision disabled, saves ~2-3 GB |
| `docker-compose.vllm-35b-1m.yaml` | vLLM | BF16 | Any | ~1M context via YaRN, util 0.95 |
| `docker-compose.sglang-35b-fp8.yaml` | SGLang | FP8 | Any | MTP speculative decoding |
| `docker-compose.llama-35b-devfix-rtx.yml` | llama.cpp | Q4_K_XL | RTX PRO 6000 | GGUF fallback (~22 GB) |
| `docker-compose.llama-35b-q4-mtp-rtx.yml` | llama.cpp + MTP | UD-Q4_K_XL | RTX PRO 6000 | Unsloth MTP-GGUF, `--spec-type mtp` — fastest 35B llama.cpp config |
| `docker-compose.llama-35b-q8-mtp-rtx.yml` | llama.cpp + MTP | Q8_0 | RTX PRO 6000 | Same as above at Q8 fidelity |
| `docker-compose.llama-35b-q4-vulkan.yml` | llama.cpp Vulkan | UD-Q4_K_XL | AMD R9700 (32 GB) | stock `server-vulkan` image, full **262 K** context, `GRAPHICS_QUEUE=1` |
| `docker-compose.llama-35b-q8-vulkan.yml` | llama.cpp Vulkan | Q8_0 | AMD ≥48 GB | DOES NOT fit on 32 GB R9700; for future ≥48 GB cards |
| `docker-compose.llama-35b-q4-mtp-vulkan.yml` | llama.cpp Vulkan + MTP | UD-Q4_K_XL | AMD R9700 (32 GB) | Custom `llama.cpp-mtp-vulkan` image (am17an PR #22673) |
| `docker-compose.llama-35b-q8-mtp-vulkan.yml` | llama.cpp Vulkan + MTP | Q8_0 | AMD ≥48 GB | Same as above at Q8; OOM on 32 GB |

## Benchmarks (RTX PRO 6000, 96 GB)

Tested 2026-04-16 with `test_chat.py` (default MoE prompt) and `test_tools.py` (4 scenarios).

### Performance

| Config | tok/s | TTFT | Think time | Engine |
|--------|------:|-----:|-----------:|--------|
| **SGLang FP8** | **204.5** | **9.2s** | 8.4s | SGLang + MTP/NEXTN speculative decoding |
| **llama.cpp Q4_K_XL** | **205.3** | **9.2s** | 8.9s | llama.cpp GGUF, single slot |
| **llama.cpp UD-Q4_K_XL + MTP** | **178.8** | **9.5s** | 9.4s | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` + `--spec-type mtp` (am17an PR #22673 build) |
| **llama.cpp Q8_0 + MTP** | **167.1** | **10.5s** | 10.5s | Same as above at Q8 fidelity |
| **vLLM FP8 text-only** | **203.2** | **7.3s** | 7.3s | vLLM v0.19.0, FP8 + vision disabled |
| **vLLM FP8** | 92.0 | 24.7s | 7.8s | vLLM v0.19.0, prefix caching |
| **vLLM BF16 text-only** | 86.6 | 27.0s | 10.2s | vLLM v0.19.0, vision disabled |
| **vLLM NVFP4 (Unsloth)** | 176.7 | 8.9s | 8.9s | vLLM cu130-nightly, `unsloth/Qwen3.6-35B-A3B-NVFP4` (group_size=8), no MTP |
| **vLLM BF16 1M** | — | — | — | OOM: BF16 weights (66 GB) leave no room for 1M KV cache on 96 GB |

SGLang's MTP speculative decoding and llama.cpp's Q4 quantization both achieve ~200+ tok/s.
vLLM without speculative decoding sits at ~90 tok/s. The 1M context BF16 config does not fit
on the RTX PRO 6000 — use FP8 or a larger GPU.

**Unsloth NVFP4 (35B-A3B) — measured 2026-05-10**: 176.7 tok/s steady-state (3 runs, `--no-think`,
runs 2–3 averaged; run 1 is cold-start at 87.7 tok/s). Within ~13 % of FP8 text-only's 203.2 tok/s
without speculative decoding. The NVFP4 quant savings barely move the needle on this MoE — only
3 B params are active per token, so per-token compute is already cheap and FP4 dequant overhead
roughly offsets the bandwidth win. Checkpoint includes MTP weights, but `--speculative-config` is
intentionally OFF (the existing 2026-04-23 finding still holds: MTP gives no speedup on 35B-A3B).

### NVIDIA vs Unsloth NVFP4 A/B + MTP (2026-06-01)

`test_scenarios.py` sweep, 5 scenarios × 3 runs each, thinking ON (compose `enable_thinking: true` overrides the CLI `--no-think`). All three runs on the same RTX PRO 6000 Blackwell host, sequential container swap, no other GPU load. vLLM 0.22.1rc1.dev30+ (vllm/vllm-openai:nightly + pip-upgrade at container start), flashinfer 0.6.12.

| Scenario | Unsloth NVFP4 | NVIDIA NVFP4 | **NVIDIA NVFP4 + MTP** | Δ MTP vs no-MTP | Δ MTP vs Unsloth |
|----------|--------------:|-------------:|-----------------------:|----------------:|-----------------:|
| Chat | 168.0 | 230.0 | **325.9** | **+42%** | **+94%** |
| RAG | 165.0 | 229.1 | **397.8** | **+74%** | **+141%** |
| Codegen | 175.4 | 241.5 | **352.3** | **+46%** | **+101%** |
| Summarization | 169.6 | 236.6 | **380.3** | **+61%** | **+124%** |
| Agentic | 174.2 | 253.1 | **382.2** | **+51%** | **+119%** |
| **Overall tok/s** | **170.4** | **238.0** | **367.7** | **+55%** | **+116%** |

TTFT also improves with MTP — ~34% lower than no-MTP NVIDIA across the board, ~52% lower than Unsloth.

Raw outputs in [benchmarks/nvfp4-ab/](benchmarks/nvfp4-ab/) (`unsloth.txt`, `nvidia.txt`, `nvidia-mtp.txt`).

**Why NVIDIA wins (no-MTP):** NVIDIA quantizes only the MoE linears to NVFP4 and keeps the rest at FP8 (`modelopt_mixed` on disk). Their per-expert layout pairs with vLLM's NVFP4 Marlin MoE kernel and flashinfer 0.6.12's `bmm_fp8_get_algos` activation-quant helper. Unsloth's pure-NVFP4 pre-fused export lands on a more conservative code path. The 40% gap is throughput-only; both quants are essentially lossless vs BF16 per their respective model cards.

**Why MTP wins (overturns 2026-04-23):** The earlier "MTP gives no speedup" finding was on the FP8 checkpoint with vLLM v0.19.x — its MTP integration was incomplete. With vLLM 0.22.1+, the NVIDIA NVFP4 checkpoint's MTP head, shared `lm_head` weights with the target model, and a triton drafter backend, MTP num=3 delivers a +55% on top of the already-fast no-MTP baseline. The "draft+verify overhead eats the gain on cheap-decode MoE" hypothesis was wrong for this combination.

**Recommendation:** use [docker-compose.vllm-35b-nvfp4-nvidia-mtp-rtx.yml](docker-compose.vllm-35b-nvfp4-nvidia-mtp-rtx.yml) (port 11438) as the default 35B endpoint.

**Loader caveat — read before deploying:** vLLM's docker-hub `:nightly` tag lags the actual wheel CDN by ~2 weeks, and the bundled flashinfer 0.6.8 is missing `bmm_fp8_get_algos`. vLLM 0.22.0 stable separately has an `lm_head.input_scale` loader bug. None of the suggested CLI flags / env vars / MoE backend choices (`marlin`, `cutlass`, `triton`, `flashinfer_*`) work around either bug. Both NVIDIA compose files fix it by pip-upgrading at container start; first boot adds ~60 s for the upgrade.

### Tool calling

| Config | Single | Parallel (2) | Chained | Multi-parallel (3) | Total |
|--------|:------:|:------------:|:-------:|:------------------:|:-----:|
| **vLLM FP8** | pass | **fail** | pass | pass | 3/4 |
| **vLLM BF16 text-only** | pass | pass | pass | pass | 4/4 |
| **SGLang FP8** | pass | pass | pass | pass | 4/4 |
| **llama.cpp Q4_K_XL** | pass | pass (1 call) | pass | pass (1 call) | 4/4* |

\* llama.cpp passes all scenarios but does not emit parallel tool calls — it dispatches
sequentially (1 call at a time), so scenarios 2 and 4 work but with warnings.

vLLM FP8 fails scenario 2 (parallel 2-city weather) by returning JSON text instead of
structured tool calls. BF16 does not have this issue. This appears to be model-level
behavior with FP8 quantization, not a vLLM bug.

### Startup time

| Config | Cold start | Notes |
|--------|:----------:|-------|
| vLLM FP8 | ~3 min | FP8 weights ~35 GB, CUDA graph capture |
| vLLM BF16 text-only | ~12 min | BF16 weights ~70 GB + Triton MoE compilation (21s torch.compile) |
| SGLang FP8 | ~3 min | FP8 weights, RadixAttention init |
| llama.cpp Q4_K_XL | ~5 min | GGUF download (~22 GB) on first run, then ~1 min |

### Known issues

- **MTP speculative decoding does not help the 35B-A3B**. Tested 2026-04-23 with
  `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}'`:
  201.7 tok/s vs 203.2 baseline (no measurable change). MoE decode on ~3B active
  params is already cheap, so the draft+verify overhead eats the speculative gain.
  The dense 27B sees +41% from the same flag because its forward pass is ~9× more
  expensive (27B vs 3B active). Both 35B FP8 configs leave MTP off by default.
- **vLLM 1M context (BF16)**: Crashes in a restart loop on 96 GB GPU. BF16 weights consume
  66 GB, leaving insufficient VRAM for 1M-token KV cache. Needs FP8 weights or >128 GB VRAM.

## Benchmarks (AMD Radeon AI PRO R9700, 32 GB)

Tested 2026-05-13 with `test_chat.py --no-think --max-tokens 256 --runs 3`
on R9700 / gfx1201, ROCm 7.2.0, Vulkan via `radv`.

| Config | tok/s | Engine |
|--------|------:|--------|
| **llama.cpp UD-Q4_K_XL · 2026-05 build · `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` · 262 K ctx** | **109.7** | Steady-state ~116. Current default in `docker-compose.llama-35b-q4-vulkan.yml`. Full native context fits — DeltaNet's KV is only ~2.7 GB at 262 K. |
| llama.cpp UD-Q4_K_XL · 2026-05 build · default env | 91.9 | Same image, no GRAPHICS_QUEUE env (+18 % from the env alone). |
| llama.cpp UD-Q4_K_XL · 2026-03 build · default env | 74.3 | Original measurement on stale image. Updating the `:server-vulkan` tag gains +23 % for free. |
| llama.cpp UD-Q4_K_XL + MTP | 62.2 | Custom `llama.cpp-mtp-vulkan` (am17an PR #22673) + GRAPHICS_QUEUE — **regresses vs stock, MoE doesn't benefit** from MTP. |
| **llama.cpp Q8_0** | OOM | 37 GB weights overflow 32 GB GDDR6 — config exists for ≥48 GB cards |
| **llama.cpp Q8_0 + MTP** | OOM | Same |

**Two optimizations that landed 75 → 109 tok/s (+46 %):**

1. **Pull the latest `ghcr.io/ggml-org/llama.cpp:server-vulkan` image** (build
   b9128 / 2026-05-13 or newer). The 2026-03 image we originally used is missing
   recent MoE coopmat tile work ([Discussion #22598](https://github.com/ggml-org/llama.cpp/discussions/22598))
   and `mat_mul_id` register-pressure fixes ([Discussion #21043](https://github.com/ggml-org/llama.cpp/discussions/21043)).
   Build-only gain: **+23 %**.
2. **`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`** routes Vulkan submissions through the
   graphics queue instead of the compute queue. On RDNA 4 MoE the graphics
   queue's lower submit latency wins by a clear margin. **+18 % on top of (1)**.
   No effect on dense 27B (measured 24.5 vs 24.1 tok/s — within noise).

What did **not** help:
- `-b 16384 -ub 2048` batch tuning — flat on top of the new image.
- `GGML_VK_DISABLE_MMVQ=1` — flat (+1 % at best).
- F16 KV cache — catastrophic regression (22 → 6 tok/s) because the fused
  Gated DeltaNet kernel falls back to CPU at f16.
- MTP — confirmed regression for 35B-A3B (same as CUDA finding).

Lucebox DFlash on 35B-A3B is **not available** — no published drafter for
the MoE variant. The 27B path (see [27B AMD section](#benchmarks-amd-radeon-ai-pro-r9700-32-gb-1) below) gets +74 % over Vulkan baseline.

See [`comparison-amd.html`](comparison-amd.html) for the full AMD comparison
with charts.

Lucebox DFlash on 35B-A3B is **not available** — no published drafter for
the MoE variant. The 27B path (see [27B AMD section](#benchmarks-amd-radeon-ai-pro-r9700-32-gb-1) below) gets +74 % over Vulkan baseline.

See [`comparison-amd.html`](comparison-amd.html) for the full AMD comparison
with charts.

## DFlash speculative decoding (experimental)

**[`docker-compose.vllm-27b-fp8-dflash-rtx.yml`](docker-compose.vllm-27b-fp8-dflash-rtx.yml)** replaces MTP with z-lab's DFlash block-diffusion drafter. Tested 2026-04-24:

| Config | Single-user tok/s | Δ vs FP8 baseline |
|--------|------------------:|-----------------:|
| FP8 baseline | 47.0 | — |
| FP8 + MTP | 66.6 | +41 % |
| **FP8 + DFlash** | **93–98** | **+99 %** |

Requirements:
- **vLLM nightly** (`vllm/vllm-openai:cu130-nightly`) — `dflash` method is NOT in v0.19.0
- **HF_TOKEN with access to gated repo** `z-lab/Qwen3.6-27B-DFlash` (accept terms on the HF page, then put token in project-root `.env`)
- Start with `docker compose --env-file ../../.env -f docker-compose.vllm-27b-fp8-dflash-rtx.yml up -d`

Caveats:
- Drafter is labelled "still under training" by z-lab — output quality is lossless in principle but monitor for regressions
- Nightly vLLM may have other regressions (untested against Gemma 4, Qwen 3.5, etc.)
- z-lab reports 4–5× on B200; we measure 2× on RTX PRO 6000 — the lower bandwidth (1.6 vs 3.4 TB/s) explains the gap
- **SGLang + MTP + DeltaNet**: Requires `--mamba-scheduler-strategy extra_buffer` and
  `SGLANG_ENABLE_SPEC_V2=1` to avoid radix cache conflict with speculative decoding.
- **FP8 KV cache**: Omitted on all configs — DeltaNet linear attention state produces
  silent corruption with FP8 KV quantization (vllm-project/vllm#26646).
- **llama.cpp parallel tool calls**: Does not support emitting multiple tool calls in a
  single response; dispatches them sequentially.

## Concurrency sweep — FP8 (35B-A3B MoE)

Full GuideLLM sweep across 5 production-shaped scenarios. See [`benchmark-35b-a3b.html`](benchmark-35b-a3b.html) for charts.

### Single-request throughput (C=1)

| Scenario | Input × Output | tok/s |
|----------|---:|---:|
| Codegen | 4 K × 1.5 K | **209.6** |
| Chat | 2 K × 300 | **196.9** |
| Agentic | 16 K × 800 | 180.6 |
| RAG | 8 K × 256 | 175.0 |
| Summarization | 12 K × 300 | 164.9 |

### Aggregate throughput & user capacity

| Scenario | Peak agg tok/s | Sweet spot C | Practical users |
|----------|---:|---:|:---:|
| Codegen | **4 166** | 2–4 | **~14** |
| Chat | **1 827** | 10–13 | **~28** |
| Agentic | 1 366 | 1–2 | ~5 |
| RAG | 809 | 3–4 | ~9 |
| Summarization | 699 | 2 | ~5 |

### 27B dense vs 35B-A3B MoE — same GPU

| Scenario | 27B C=1 | 35B C=1 | Speedup | 27B peak agg | 35B peak agg | Speedup |
|----------|---:|---:|---:|---:|---:|---:|
| Chat | 66.6 | 196.9 | 2.96× | 793 | 1 827 | 2.30× |
| Codegen | 69.1 | 209.6 | 3.03× | 1 094 | 4 166 | 3.81× |
| Agentic | 54.1 | 180.6 | 3.34× | 147 | 1 366 | 9.29× |
| RAG | 50.7 | 175.0 | 3.45× | 230 | 809 | 3.52× |
| Summarization | 46.0 | 164.9 | 3.58× | 188 | 699 | 3.72× |

MoE 35B is the clear throughput winner — ~3× decode and 3–9× aggregate. Pick it for multi-user serving. Pick the dense 27B only when per-token reasoning quality trumps throughput.

Reproduce with `TARGET=http://localhost:11435 MODEL=qwen3.6-35b PROCESSOR=Qwen/Qwen3.6-35B-A3B-FP8 OUTPUT_DIR=$(pwd)/benchmarks/guidellm-35b ./bench-guidellm-parallel.sh`.

## Testing

Run the shared test scripts to verify the endpoint:

```bash
# Chat benchmark — measures TTFT and tok/s
python ../shared/test_chat.py --model qwen3.6-35b

# Tool-calling integration test — single, parallel, chained, multi-parallel
python ../shared/test_tools.py --model qwen3.6-35b
```

## Requirements

- **vLLM**: v0.19.0+ (`vllm/vllm-openai:v0.19.0-cu130`)
- **SGLang**: v0.5.10+ (`lmsysorg/sglang:latest`)
- **llama.cpp**: latest `server-cuda` image

---

# Qwen3.6-27B

[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) — 27B dense model with the same hybrid attention architecture as the 35B MoE. First open-weight single-dense variant of the Qwen3.6 family.

## Architecture

- **Dense 27B**: all 27B parameters active per token (vs 3B for 35B-A3B MoE)
- **Hybrid attention**: 64 layers = 16 × (3×Gated DeltaNet + 1×full Gated Attention) — 48 DeltaNet + 16 full
- **Multimodal**: text, image, and video input (vision encoder optional)
- **Context**: 262K native, 1M via YaRN
- **Multi-Token Prediction**: native MTP for speculative decoding
- **License**: Apache 2.0

## Quick start

```bash
# RTX PRO 6000 (96 GB) — primary config
docker compose -f docker-compose.vllm-27b-fp8-rtx.yml up -d
```

API endpoint: `http://localhost:11436/v1`, model name: `qwen3.6-27b`

## Compose variants

| File | Engine | Quant | Hardware | Notes |
|------|--------|-------|----------|-------|
| `docker-compose.vllm-27b-fp8-rtx.yml` | vLLM | FP8 | RTX PRO 6000 | Primary config, 262K context |
| `docker-compose.vllm-27b-fp8-textonly-rtx.yml` | vLLM | FP8 | RTX PRO 6000 | FP8 + vision disabled |
| `docker-compose.vllm-27b-nvfp4-rtx.yml` | vLLM | NVFP4 | RTX PRO 6000 | Community NVFP4 (mmangkad/Qwen3.6-27B-NVFP4) |
| `docker-compose.vllm-27b-nvfp4-unsloth-rtx.yml` | vLLM | NVFP4 | RTX PRO 6000 | Unsloth NVFP4 (group_size=8); no MTP heads |
| `docker-compose.vllm-27b-text-only.yaml` | vLLM | BF16 | Any | Vision disabled, ~54 GB BF16 weights |
| `docker-compose.sglang-27b-fp8.yaml` | SGLang | FP8 | Any | MTP speculative decoding |
| `docker-compose.llama-27b-q4-mtp-rtx.yml` | llama.cpp + MTP | UD-Q4_K_XL | RTX PRO 6000 | Unsloth MTP-GGUF, `--spec-type mtp` — best 27B llama.cpp config |
| `docker-compose.llama-27b-q8-mtp-rtx.yml` | llama.cpp + MTP | Q8_0 | RTX PRO 6000 | Same as above at Q8 fidelity |
| `docker-compose.llama-27b-q4-vulkan.yml` | llama.cpp Vulkan | UD-Q4_K_XL | AMD R9700 (32 GB) | stock `server-vulkan` image, 64K context |
| `docker-compose.llama-27b-q8-vulkan.yml` | llama.cpp Vulkan | Q8_0 | AMD R9700 (32 GB, tight) | 16K context to fit alongside KV cache |
| `docker-compose.llama-27b-q4-mtp-vulkan.yml` | llama.cpp Vulkan + MTP | UD-Q4_K_XL | AMD R9700 (32 GB) | Custom `llama.cpp-mtp-vulkan` image (am17an PR #22673) |
| `docker-compose.llama-27b-q8-mtp-vulkan.yml` | llama.cpp Vulkan + MTP | Q8_0 | AMD R9700 (32 GB, tight) | Same as above at Q8 |
| `docker-compose.dflash-27b-q4-rocm.yml` | Lucebox DFlash (HIP) | Q4_K_M | AMD R9700 (gfx1201, EXPERIMENTAL) | PR #119 + #159; tiles tuned for gfx1151 upstream |
| `docker-compose.vllm-27b-int4-rtx.yml` | vLLM | INT4 AutoRound + MTP | RTX PRO 6000 | `Lorbus/Qwen3.6-27B-int4-AutoRound`, 16.9 GiB |
| `docker-compose.vllm-27b-int4-dflash-rtx.yml` | vLLM | INT4 + DFlash N=8 | RTX PRO 6000 | **Champion** — 146 tok/s steady state |
| `docker-compose.vllm-27b-int4-baseline-rtx.yml` | vLLM | INT4 (no spec) | RTX PRO 6000 | Quant-only reference |
| `docker-compose.vllm-27b-nvfp4-mtp-rtx.yml` | vLLM | NVFP4 + MTP | RTX PRO 6000 | `sakamakismile` NVFP4-MTP, group=16 |
| `docker-compose.vllm-27b-nvfp4-baseline-rtx.yml` | vLLM | NVFP4 (no spec) | RTX PRO 6000 | Quant-only reference |
| `docker-compose.vllm-27b-nvfp4-nvidia-rtx.yml` | vLLM | NVFP4 (modelopt_mixed) | RTX PRO 6000 | **Official NVIDIA NVFP4** (`nvidia/Qwen3.6-27B-NVFP4`, modelopt v0.45). 71.1 tok/s — fastest no-MTP NVFP4, +58% over community mmangkad. Entrypoint pip-upgrades vLLM nightly + flashinfer on boot (same recipe as 35B NVIDIA) |
| `docker-compose.vllm-27b-nvfp4-nvidia-mtp-rtx.yml` | vLLM | NVFP4 + MTP | RTX PRO 6000 | NVIDIA NVFP4 + `qwen3_next_mtp` N=1 → 95.9 tok/s (+35%). Same loader recipe, port 11436 |
| `docker-compose.vllm-27b-nvfp4-nvidia-mtp-parallel-rtx.yml` | vLLM | NVFP4 + MTP | RTX PRO 6000 | Concurrency-tuned (`--max-num-seqs 256`) NVIDIA NVFP4+MTP for the parallelization study; wins long-output workloads at scale |
| `docker-compose.vllm-27b-fp8-mtp-parallel-rtx.yml` | vLLM | FP8 + MTP | RTX PRO 6000 | Concurrency-tuned FP8+MTP A/B partner; wins high-concurrency short-reply serving |

## Benchmarks (RTX PRO 6000, 96 GB)

Tested 2026-04-22 with `test_chat.py` (default MoE prompt).

### Performance

| Config | tok/s | TTFT | Think time | Engine |
|--------|------:|-----:|-----------:|--------|
| **vLLM INT4 + DFlash N=8** | **140.2 / 146 steady** | **13.2s** | — | vLLM cu130-nightly, `Lorbus/Qwen3.6-27B-int4-AutoRound` + `z-lab/Qwen3.6-27B-DFlash` ★ champion |
| **vLLM INT4 + MTP** | **102.8** | — | — | vLLM cu130-nightly, AutoRound INT4 + `qwen3_next_mtp` N=1 |
| **vLLM FP8 + DFlash** | 103.0 | 19.2s | — | vLLM cu130-nightly, `z-lab/Qwen3.6-27B-DFlash` drafter |
| **vLLM NVFP4 + MTP (NVIDIA official)** | **95.9** | — | — | vLLM 0.24.0, `nvidia/Qwen3.6-27B-NVFP4` (modelopt v0.45) + `qwen3_next_mtp` N=1 — fastest NVFP4 (2026-06-30) |
| **llama.cpp UD-Q4_K_XL + MTP** | **85.7** | **18.9s** | 18.7s | `unsloth/Qwen3.6-27B-MTP-GGUF` + `--spec-type mtp` (am17an PR #22673 build) |
| **vLLM NVFP4 + MTP (sakamakismile)** | **83.5** | — | — | vLLM cu130-nightly, `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (group_size=16) + `qwen3_next_mtp` N=1 |
| **vLLM INT4 baseline** | 80.7 | 21.7s | — | INT4 weights alone, no spec — quant-only gain |
| **vLLM NVFP4 (NVIDIA official, no MTP)** | **71.1** | — | — | vLLM 0.24.0, `nvidia/Qwen3.6-27B-NVFP4` (modelopt v0.45) — +58% over community mmangkad (44.9), best no-MTP NVFP4 (2026-06-30) |
| **vLLM FP8 + MTP** | 66.2 | 26.0s | 26.0s | vLLM v0.19.0, FP8 + `qwen3_next_mtp` speculative decoding |
| **llama.cpp Q8_0 + MTP** | **66.1** | **26.3s** | 26.2s | `unsloth/Qwen3.6-27B-MTP-GGUF` Q8_0 + `--spec-type mtp` |
| **vLLM NVFP4 (sakamakismile, MTP dormant)** | 57.4 | — | — | vLLM cu130-nightly, same checkpoint as row 1 with `--speculative-config` omitted (MTP weights present but unused) |
| **vLLM NVFP4 (Unsloth)** | 49.7 | 37.8s | 37.7s | vLLM cu130-nightly, `unsloth/Qwen3.6-27B-NVFP4` (group_size=8), no MTP |
| **vLLM FP8 text-only** | 47.0 | 35.4s | 35.3s | vLLM v0.19.0, FP8 + vision disabled (no MTP) |
| **vLLM FP8** | 47.0 | 41.1s | 41.0s | vLLM v0.19.0, FP8 multimodal (no MTP) |
| **vLLM NVFP4 (mmangkad)** | 44.9 | 36.7s | 36.6s | vLLM v0.19.0, FlashInfer Cutlass NVFP4 backend, single GPU |
| **vLLM BF16 text-only** | 28.3 | 64.9s | 64.8s | vLLM v0.19.0, BF16 + vision disabled |
| **SGLang FP8 + MTP** | **72.1** | 23.0s | 23.0s | `lmsysorg/sglang:latest` NEXTN spec decoding — **now produces clean output** (verified 2026-06-29); the earlier "garbage / EAGLE fallback" bug is fixed. ~9% over vLLM FP8+MTP single-stream; tools 4/4 |

Dense 27B decode is bandwidth-bound on all 27B active params (vs 3B for 35B-MoE):
~1.6 TB/s ÷ 27 GB ≈ 60 tok/s theoretical peak at FP8. Measured 47 tok/s is ~78%
efficiency, typical for FP8 tensor-core kernels on Blackwell SM_120.

BF16 weights (~54 GB) are 2× the size, so decode drops to 28 tok/s — exactly
halved, confirming the bandwidth-bound profile.

The vision encoder adds negligible overhead for text-only inference (both FP8
variants measured identical throughput).

**NVFP4 surprise**: Despite Blackwell's hardware NVFP4 tensor cores, single-GPU
NVFP4 measured 44.9 tok/s — *slightly slower* than FP8. Why: only the linear
ops in transformer blocks are FP4-quantized (~14 GB), the rest stays BF16/FP8
(~17 GB), so total memory bandwidth pressure is similar to plain FP8. The
upstream community recipe achieves higher throughput by adding `--tensor-parallel-size 2`
to split weights across two GPUs — which is the actual source of speedup,
not NVFP4 itself. NVFP4 may be more advantageous for prefill or multi-GPU setups.

**Unsloth NVFP4 — measured 2026-05-10**: 49.7 tok/s steady-state (3 runs, `--no-think`,
runs 2–3 averaged; run 1 is cold-start at 37.3 tok/s).
[`unsloth/Qwen3.6-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) ships with
group_size=8 (vs 16 in mmangkad/sakamakismile variants) — a finer-grained scale that nudges
quality up at the cost of slightly more dequant metadata per block. Throughput sits ~5 tok/s
above the mmangkad NVFP4 baseline (44.9) and within run-to-run noise of FP8 (47.0). **No MTP
heads** in this checkpoint, so speculative decoding is off; the same checkpoint cannot reach
the 83.7 tok/s figure the carnice-v2 sister repo gets, because that recipe relies on
`sakamakismile/Carnice-V2-27b-NVFP4-TEXT-MTP` which bakes MTP weights into the safetensors.
To get +60 % on the unsloth weights you'd need to pair them with an external drafter
(z-lab DFlash) or switch to the sakamakismile MTP-bundled checkpoint.

**MTP speculative decoding wins big**: Adding
`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}'`
to the FP8 config jumped throughput from 47 → 66.2 tok/s on long generations
(+41%). On the NVFP4 path the lift is even larger — 44.9 → 83.5 tok/s (+86%),
because the slower per-step decode leaves more headroom for the speculative
batched verification to amortize the draft+verify overhead. MTP uses Qwen3.6's
native draft heads to speculate one token ahead; verification batches the
draft + target in a single forward pass. Pure-decode on short outputs
(~64 tokens) dropped slightly (-13%) due to draft+verify overhead, but long
generations dominate this thinking model's typical workload.
`num_speculative_tokens=2` was untested — DeltaNet hybrid recurrent state
can't restore on partial accept, so gains saturate near 1.

**MTP requires baked-in head weights, not just any NVFP4 checkpoint.** As of
2026-06 there are **four** 27B NVFP4 quants; **three** ship the MTP layer —
Unsloth **re-uploaded** its checkpoint (now group_size=16 *with* an MTP head; it
was group_size=8 / no-MTP when first measured 2026-05-10), and NVIDIA's official
checkpoint landed 2026-06-30 with the native MTP head intact:

| Checkpoint | group_size | MTP head | Best measured |
|---|---:|:---:|---:|
| `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | 16 | yes | **98.4 tok/s** (v0.23.0 + MTP; 83.5 on cu130-nightly) |
| `nvidia/Qwen3.6-27B-NVFP4` (official, modelopt v0.45) | 16 | yes | **95.9 tok/s** (vLLM 0.24.0 + MTP; **71.1 no-MTP** — fastest no-MTP NVFP4) |
| `unsloth/Qwen3.6-27B-NVFP4` | 16 | yes (since ~2026-06) | 81.4 tok/s (v0.23.0 + MTP; was 49.7 / no-MTP) |
| `mmangkad/Qwen3.6-27B-NVFP4` | 16 | no | 44.9 tok/s (no spec; also forces fp8 KV → vLLM rejects DFlash combo) |

Unsloth's MTP head underperforms sakamakismile's by ~21% (81.4 vs 98.4) — same
pattern as the 35B NVIDIA-vs-Unsloth result; see the v0.23.0 re-test below.

**NVIDIA official NVFP4 (measured 2026-06-30, vLLM 0.24.0):** the official
`nvidia/Qwen3.6-27B-NVFP4` (modelopt v0.45, `modelopt_mixed`: FFN linears at
W4A16_NVFP4 group=16, attention at FP8, lm_head 4-bit) is the **fastest no-MTP
NVFP4** at 71.1 tok/s — +58% over the community mmangkad baseline (44.9) and
above FP8 (47). Its native MTP head via `qwen3_next_mtp` N=1 lifts that to 95.9
tok/s (+35%), within ~3% of sakamakismile's MTP checkpoint and second only to
the INT4/DFlash speculative configs. Served via the same pip-upgrade loader
recipe as the 35B NVIDIA config (docker-hub `:nightly` lags the wheel CDN). Note:
vLLM auto-selects `kv_cache_dtype=fp8_e4m3` from the checkpoint's
`kv_cache_quant_algo=fp8`; output stayed coherent on vLLM 0.24.0 (no #26646
corruption observed), but if quality regresses, this is the first thing to check.
Raw runs in [benchmarks/nvfp4-nvidia-27b.txt](benchmarks/nvfp4-nvidia-27b.txt).

For a no-MTP NVFP4 checkpoint to match 83.5, you'd need to pair it with an
external drafter (e.g. z-lab DFlash) — see the experimental DFlash section
below.

The Blackwell flag bundle (`--async-scheduling`, `--cuda-graph-capture-size`,
`--compilation-config '{"level":3}'`) suggested by community blogs is
**redundant or invalid** in vLLM v0.19: async scheduling is on by default,
the cuda-graph CLI flag doesn't exist (it's a `compilation-config` JSON
sub-field), and `level:3` (VLLM_COMPILE) is the default `mode`. No gains
available from this path.

### vLLM v0.23.0 re-test — 3-way A/B/C (2026-06-15)

Triggered by the [unsloth/Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4)
**re-upload** (it now ships an MTP head, see above) and vLLM **v0.23.0**'s NVFP4
linear refactor + EAGLE/MTP lookahead-cache work. Engine held constant at
`vllm/vllm-openai:v0.23.0-cu129` (no cu130 tag exists for 0.23.0; cu129 = CUDA
12.9, supports Blackwell SM_120). `test_chat.py --runs 3 --warmup --no-think`,
runs averaged. Script: [`benchmarks/run_3way_v0230.sh`](benchmarks/run_3way_v0230.sh).

| Config | Checkpoint | Spec | avg tok/s | Range | VRAM | Tools |
|--------|-----------|------|----------:|-------|------|:-----:|
| **B FP8 + DFlash** | `Qwen3.6-27B-FP8` + `z-lab DFlash` (15 tok) | DFlash | **99.7** | 86.7–106.4 (high variance) | ~27 GB FP8 | — |
| **C NVFP4 + MTP (sakamakismile)** | `sakamakismile/…-NVFP4-MTP` | MTP N=1 | **98.4** | 97.6–99.4 (tight) | ~16 GB NVFP4 | — |
| **A NVFP4 + MTP (unsloth)** | `unsloth/Qwen3.6-27B-NVFP4` | MTP N=1 | **81.4** | 80.2–82.4 | ~16 GB NVFP4 | **1/1 PASS** |

**Takeaways**

- **v0.23.0 helps the NVFP4+MTP path: sakamakismile 83.5 → 98.4 tok/s (+18%)** vs
  the 2026-04-22 cu130-nightly number, same checkpoint/flags. FP8+DFlash is flat
  (~103 → 99.7, within DFlash's run-to-run variance).
- **The Unsloth re-upload is a real gain for that checkpoint** (49.7 no-MTP →
  81.4 with its new MTP head) **but it is still the slowest MTP option** —
  sakamakismile beats it by +21% at the same quant/VRAM/settings. Unsloth's MTP
  head simply has lower acceptance. No reason to switch the live model to it.
- **Best VRAM-for-throughput: C (sakamakismile NVFP4+MTP)** — 98.4 tok/s, tight
  variance, at **~16 GB** vs FP8+DFlash's 99.7 tok/s at ~27 GB. Near-identical
  speed, ~40% less VRAM, and steadier. The standout candidate if memory headroom
  matters (longer ctx, or freeing the card).
- **Tool-calling** on the unsloth candidate passed (`single` scenario: `get_weather`
  → `tool_calls`), confirming NVFP4+MTP tool use works under v0.23.0.

**⚠️ v0.23.0 migration gotcha (breaking change).** v0.23.0 tightened config
validation and now **hard-fails** instead of silently clamping:

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (1019).
```

Every Qwen3.6 hybrid-DeltaNet compose needs `--max-num-seqs ≤ Mamba-cache-blocks`
(~1019 at util 0.90) added when moving off cu130-nightly. Fixed here by adding
`--max-num-seqs 512`. Only the unsloth compose is pinned to v0.23.0-cu129 (it
needs the native NVFP4+MTP path); FP8+DFlash and sakamakismile composes remain
on cu130-nightly — bump them similarly only with the max-num-seqs flag.

### Startup time

| Config | Cold start | Notes |
|--------|:----------:|-------|
| vLLM FP8 | ~3.5 min | FP8 weights ~27 GB, CUDA graph capture |
| vLLM FP8 text-only | ~3.5 min | Same as FP8, vision encoder still loads |
| SGLang FP8 | ~2 min | Fast startup; MTP draft heads initialize quickly |
| vLLM BF16 text-only | ~12 min | BF16 weights ~54 GB + Triton compilation |

### Known issues

- **No GGUF**: community quants not yet published for 27B.
- **FP8 KV cache**: Omitted on all configs — DeltaNet linear attention state
  produces silent corruption with FP8 KV quantization (vllm-project/vllm#26646).

## Benchmarks (AMD Radeon AI PRO R9700, 32 GB)

Tested 2026-05-13 with `test_chat.py --no-think --max-tokens 256 --runs 3`
on R9700 / gfx1201, ROCm 7.2.0, kernel 6.17, Vulkan via `radv`. Full charts
and the corresponding 35B-A3B numbers in [`comparison-amd.html`](comparison-amd.html).

| Config | tok/s | TTFT | Engine |
|--------|------:|-----:|--------|
| **Lucebox DFlash (HIP)** | **38.0** | **500 ms** | `lucebox-dflash:latest` (PR #119 + #159 built for gfx1201), Q4_K_M + DFlash drafter Q8_0 |
| **llama.cpp UD-Q4_K_XL + MTP** | **26.1** | nan† | `llama.cpp-mtp-vulkan:latest` (am17an PR #22673 built with `-DGGML_VULKAN=ON`), `--spec-type draft-mtp`, `--spec-draft-n-max 3` |
| **llama.cpp UD-Q4_K_XL · 2026-05 build (stock)** | **24.5** | nan† | `ghcr.io/ggml-org/llama.cpp:server-vulkan` b9128+, 65K ctx, `-fa on`, KV q8_0 |
| llama.cpp UD-Q4_K_XL · 2026-03 build | 21.9 | ~900 ms | Earlier stale-image measurement; +13 % from pulling latest |
| **llama.cpp Q8_0 (stock)** | **15.3** | ~810 ms | Same image, 16K ctx — Q8 weights ~28.6 GB just fit with KV cache |

`GGML_VK_ALLOW_GRAPHICS_QUEUE=1` (the 35B-A3B winner) is **not** added to
27B compose files — empirically flat on dense 27B (24.5 → 24.1 tok/s, noise).

† MTP runs return `usage.completion_tokens=256` with TTFT not surfaced in
the streamed reasoning protocol; tok/s is wall-clock based and reliable.

**DFlash wins on R9700.** PR #119's HIP port of the rocWMMA flashprefill
kernels compiles cleanly for gfx1201 (RDNA 4) despite being tuned upstream
for gfx1100/gfx1151. End-to-end at 256-token decode it's 1.74× the stock
Vulkan baseline and 1.46× over MTP-Vulkan. Compare to amd.md's 26.85 tok/s
on Strix Halo (gfx1151 + LPDDR5X-8000) — the R9700's GDDR6 bandwidth
(~640 GB/s vs ~256 GB/s) explains the lift.

**MTP on Vulkan works.** The am17an PR #22673 branch builds with
`-DGGML_VULKAN=ON` once `spirv-headers` is added to the builder image
(see [`Dockerfile.llama-mtp-vulkan`](Dockerfile.llama-mtp-vulkan)). Flag
name changed since the CUDA branch was wired up: `--spec-type mtp` →
`--spec-type draft-mtp`. The `+19%` lift (21.9 → 26.1) is smaller than the
CUDA Q4 + MTP gain on the dense 27B (47 → 85.7, +82%) — Vulkan's lower
draft-batch parallelism leaves less room for spec verification to amortize.

**Known issues on AMD/Vulkan**:
- `radv: not a conformant Vulkan implementation, testing use only.` —
  Mesa's RADV driver reports as testing-only but works correctly for our
  workload. Use AMDVLK if RADV regresses.
- DFlash bench on Strix Halo (amd.md) used `Qwen3.6-27B-Q4_K_M.gguf` from
  the non-MTP repo plus the Lucebox drafter. We use the same combination
  for parity; the MTP-GGUF variant is **not** the right target for DFlash
  (DFlash uses its own drafter, not the bundled MTP heads).
- 35B-Q8 and 35B-Q8-MTP compose files exist but **OOM on 32 GB** (weights
  alone are ~37 GB). Included for future ≥48 GB AMD cards.

## Testing

```bash
# Chat benchmark
python ../shared/test_chat.py --base-url http://localhost:11436/v1 --model qwen3.6-27b

# Tool-calling integration test
python ../shared/test_tools.py --base-url http://localhost:11436/v1 --model qwen3.6-27b
```

## When to pick 27B vs 35B-A3B

- **35B-A3B (MoE)**: ~4× faster decode (200+ tok/s vs 47 tok/s). Best for high-throughput, multi-user, or latency-sensitive serving.
- **27B (dense)**: stronger per-token reasoning and generation quality. Best for single-user agentic tasks where quality matters more than speed.

## Concurrency sweep — FP8 + MTP

Full GuideLLM sweep across 5 production-shaped scenarios. See [`benchmark-27b.html`](benchmark-27b.html) for charts.

### Single-request throughput (C=1)

| Scenario | Input × Output | tok/s |
|----------|---:|---:|
| Codegen | 4 K × 1.5 K | **69.1** |
| Chat | 2 K × 300 | **66.6** |
| Agentic | 16 K × 800 | 54.1 |
| RAG | 8 K × 256 | 50.7 |
| Summarization | 12 K × 300 | 46.0 |

### Aggregate throughput & user capacity

| Scenario | Peak agg tok/s | Sweet spot C | Practical concurrent users |
|----------|---:|---:|:---:|
| Chat | **793** | 8–12 | **~15** |
| Codegen | **1 094** | 4–6 | ~6 |
| RAG | 230 | 2–3 | ~3 |
| Summarization | 188 | 1 | ~2 |
| Agentic | 147 | 1 | 1 |

### Key findings

- **MTP scales well under load** — chat throughput stays positive through C=15 (no compounding overhead).
- **Bottleneck is prefill, not decode** — long-prompt scenarios (RAG, agentic, summarization) cap at 1–3 concurrent users; the "60 tok/s decode ceiling" is rarely the binding constraint when prompts grow.
- **One RTX PRO 6000 serves**: 15 chat users · 6 coders · 3 RAG users · 1 agentic agent.

Reproduce with `./bench-guidellm-parallel.sh` (defaults to vLLM FP8+MTP on port 11436).

## Parallelization: NVFP4 vs FP8 (both + MTP), 2026-07-01

Head-to-head of the two official checkpoints **both with MTP** (`qwen3_next_mtp` N=1) under a
fixed-concurrency ladder [1, 4, 8, 16, 32], four production-shaped scenarios. Charts +
methodology in [`comparison-nvfp4-fp8-parallel.html`](comparison-nvfp4-fp8-parallel.html); raw
guidellm output under [`benchmarks/parallel-2x2/`](benchmarks/parallel-2x2/). Configs:
[`docker-compose.vllm-27b-nvfp4-nvidia-mtp-parallel-rtx.yml`](docker-compose.vllm-27b-nvfp4-nvidia-mtp-parallel-rtx.yml)
and [`docker-compose.vllm-27b-fp8-mtp-parallel-rtx.yml`](docker-compose.vllm-27b-fp8-mtp-parallel-rtx.yml)
(both text-only, `--max-num-seqs 256`, `--max-model-len 32768`, `--gpu-memory-utilization 0.90`).

Aggregate output tok/s — single-stream vs saturated:

| Scenario (in/out) | C=1 NVFP4 | C=1 FP8 | peak NVFP4 | peak FP8 | winner @ scale |
|---|---:|---:|---:|---:|---|
| chat (2K/300)     | **87.3** | 68.2 | 418 @c26 | **677 @c29** | FP8 +62% |
| rag (8K/256)      | **64.0** | 51.7 | 148 @c13 | **198 @c21** | FP8 +34% |
| agentic (16K/800) | **80.0** | 54.9 | **211 @c6** | 180 @c6 | NVFP4 +17% |
| codegen (4K/1.5K) | **102.8** | 71.9 | **811 @c19** | 704 @c19 | NVFP4 +15% |

**Findings (these overturned the going-in hypothesis):**

1. **NVFP4+MTP wins single-stream in every scenario** (C=1, +20–40%). For 1–4 concurrent users it is strictly better and is the recommended default endpoint.
2. **Output length decides the parallel winner, not the quant.** Short-output workloads (chat 300, rag 256) go **compute-bound** when batched, where NVFP4's group=16 dequant tax dominates and FP8's W8A8 kernels pull ahead past a crossover around C≈5. Long-output workloads (codegen 1.5K, agentic 800) stay **bandwidth-bound**, where MTP + 4-bit weights keep NVFP4 in front throughout.
3. **NVFP4's ~2.1× KV-cache ceiling is a mirage here.** vLLM reports 39.8× vs 19.2× max concurrency at 32K/req (61.4 vs 49.8 GiB free; NVFP4 also gets fp8 KV from the modelopt checkpoint), but **compute saturates before KV does** — on rag, FP8 actually sustained *higher* concurrency (21 vs 15) and NVFP4's throughput fell at the top step. The KV headroom would only pay off on very long shared-prefix contexts where KV, not compute, is the wall.

**Recommendation:** NVFP4+MTP for low-concurrency or long-generation (coding/agents); FP8+MTP for high-concurrency short-reply serving (chat, RAG Q&A at 10+ users).

**Confound:** the FP8 checkpoint runs vLLM v0.19.0 (stable image) and NVFP4 v0.24.0 (nightly, required by the modelopt loader) — a version difference on the batched-compute kernels, not purely the quant. Reproduce with `./bench-guidellm-concurrent.sh` (fixed-concurrency ladder; the sweep script's throughput strategy dispatches 512 requests and yields zero completions for long scenarios).
