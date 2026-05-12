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
| `docker-compose.vllm-35b-nvfp4-unsloth-rtx.yml` | vLLM | NVFP4 | RTX PRO 6000 | Unsloth NVFP4 (group_size=8); no MTP active |
| `docker-compose.vllm-35b-text-only.yaml` | vLLM | BF16 | Any | Vision disabled, saves ~2-3 GB |
| `docker-compose.vllm-35b-1m.yaml` | vLLM | BF16 | Any | ~1M context via YaRN, util 0.95 |
| `docker-compose.sglang-35b-fp8.yaml` | SGLang | FP8 | Any | MTP speculative decoding |
| `docker-compose.llama-35b-devfix-rtx.yml` | llama.cpp | Q4_K_XL | RTX PRO 6000 | GGUF fallback (~22 GB) |
| `docker-compose.llama-35b-q4-mtp-rtx.yml` | llama.cpp + MTP | UD-Q4_K_XL | RTX PRO 6000 | Unsloth MTP-GGUF, `--spec-type mtp` — fastest 35B llama.cpp config |
| `docker-compose.llama-35b-q8-mtp-rtx.yml` | llama.cpp + MTP | Q8_0 | RTX PRO 6000 | Same as above at Q8 fidelity |

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
| `docker-compose.vllm-27b-int4-rtx.yml` | vLLM | INT4 AutoRound + MTP | RTX PRO 6000 | `Lorbus/Qwen3.6-27B-int4-AutoRound`, 16.9 GiB |
| `docker-compose.vllm-27b-int4-dflash-rtx.yml` | vLLM | INT4 + DFlash N=8 | RTX PRO 6000 | **Champion** — 146 tok/s steady state |
| `docker-compose.vllm-27b-int4-baseline-rtx.yml` | vLLM | INT4 (no spec) | RTX PRO 6000 | Quant-only reference |
| `docker-compose.vllm-27b-nvfp4-mtp-rtx.yml` | vLLM | NVFP4 + MTP | RTX PRO 6000 | `sakamakismile` NVFP4-MTP, group=16 |
| `docker-compose.vllm-27b-nvfp4-baseline-rtx.yml` | vLLM | NVFP4 (no spec) | RTX PRO 6000 | Quant-only reference |

## Benchmarks (RTX PRO 6000, 96 GB)

Tested 2026-04-22 with `test_chat.py` (default MoE prompt).

### Performance

| Config | tok/s | TTFT | Think time | Engine |
|--------|------:|-----:|-----------:|--------|
| **vLLM INT4 + DFlash N=8** | **140.2 / 146 steady** | **13.2s** | — | vLLM cu130-nightly, `Lorbus/Qwen3.6-27B-int4-AutoRound` + `z-lab/Qwen3.6-27B-DFlash` ★ champion |
| **vLLM INT4 + MTP** | **102.8** | — | — | vLLM cu130-nightly, AutoRound INT4 + `qwen3_next_mtp` N=1 |
| **vLLM FP8 + DFlash** | 103.0 | 19.2s | — | vLLM cu130-nightly, `z-lab/Qwen3.6-27B-DFlash` drafter |
| **llama.cpp UD-Q4_K_XL + MTP** | **85.7** | **18.9s** | 18.7s | `unsloth/Qwen3.6-27B-MTP-GGUF` + `--spec-type mtp` (am17an PR #22673 build) |
| **vLLM NVFP4 + MTP (sakamakismile)** | **83.5** | — | — | vLLM cu130-nightly, `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (group_size=16) + `qwen3_next_mtp` N=1 |
| **vLLM INT4 baseline** | 80.7 | 21.7s | — | INT4 weights alone, no spec — quant-only gain |
| **vLLM FP8 + MTP** | 66.2 | 26.0s | 26.0s | vLLM v0.19.0, FP8 + `qwen3_next_mtp` speculative decoding |
| **llama.cpp Q8_0 + MTP** | **66.1** | **26.3s** | 26.2s | `unsloth/Qwen3.6-27B-MTP-GGUF` Q8_0 + `--spec-type mtp` |
| **vLLM NVFP4 (sakamakismile, MTP dormant)** | 57.4 | — | — | vLLM cu130-nightly, same checkpoint as row 1 with `--speculative-config` omitted (MTP weights present but unused) |
| **vLLM NVFP4 (Unsloth)** | 49.7 | 37.8s | 37.7s | vLLM cu130-nightly, `unsloth/Qwen3.6-27B-NVFP4` (group_size=8), no MTP |
| **vLLM FP8 text-only** | 47.0 | 35.4s | 35.3s | vLLM v0.19.0, FP8 + vision disabled (no MTP) |
| **vLLM FP8** | 47.0 | 41.1s | 41.0s | vLLM v0.19.0, FP8 multimodal (no MTP) |
| **vLLM NVFP4 (mmangkad)** | 44.9 | 36.7s | 36.6s | vLLM v0.19.0, FlashInfer Cutlass NVFP4 backend, single GPU |
| **vLLM BF16 text-only** | 28.3 | 64.9s | 64.8s | vLLM v0.19.0, BF16 + vision disabled |
| **SGLang FP8 + MTP** | — | — | — | Config present but NEXTN spec decoding produces garbage (EAGLE fallback); fix pending |

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

**MTP requires baked-in head weights, not just any NVFP4 checkpoint.** Three
27B NVFP4 quants exist; only one ships the MTP layer in the safetensors:

| Checkpoint | group_size | MTP head | Best measured |
|---|---:|:---:|---:|
| `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | 16 | yes | **83.5 tok/s** (+ MTP) |
| `unsloth/Qwen3.6-27B-NVFP4` | 8 | no | 49.7 tok/s (no spec) |
| `mmangkad/Qwen3.6-27B-NVFP4` | 16 | no | 44.9 tok/s (no spec; also forces fp8 KV → vLLM rejects DFlash combo) |

For a no-MTP NVFP4 checkpoint to match 83.5, you'd need to pair it with an
external drafter (e.g. z-lab DFlash) — see the experimental DFlash section
below.

The Blackwell flag bundle (`--async-scheduling`, `--cuda-graph-capture-size`,
`--compilation-config '{"level":3}'`) suggested by community blogs is
**redundant or invalid** in vLLM v0.19: async scheduling is on by default,
the cuda-graph CLI flag doesn't exist (it's a `compilation-config` JSON
sub-field), and `level:3` (VLLM_COMPILE) is the default `mode`. No gains
available from this path.

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
