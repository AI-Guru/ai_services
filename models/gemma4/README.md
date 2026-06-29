# Gemma 4 Self-Hosted Inference

Runs Google Gemma 4 models via llama.cpp or vLLM on NVIDIA GPUs, exposing an OpenAI-compatible API.

Released April 2, 2026. First Gemma family under Apache 2.0 license. Built from Gemini 3 research.

## Models

| Model | Architecture | Total | Active | Context | Modalities |
|-------|-------------|-------|--------|---------|------------|
| **E2B** | Dense | 5.1B | 2.3B | 128K | Text, Image, Audio |
| **E4B** | Dense | 8B | 4.5B | 128K | Text, Image, Audio |
| **12B** | Dense | 12B | 12B | 128K | Text, Image, Audio |
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
| 11452 | Gemma 4 26B-A4B (llama.cpp) |
| 11453 | Gemma 4 31B |
| 11454 | Gemma 4 26B-A4B (vLLM) |
| 11455 | Gemma 4 12B (vLLM) |
| 11456 | Gemma 4 26B-A4B BF16 (SGLang — host-RAM-blocked) |
| 11457 | Gemma 4 26B-A4B NVFP4 (SGLang) |

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
├─ Mid-size dense (between E4B and 26B)?
│  └─ vLLM BF16 ───────────── vllm-12b-bf16-rtx.yml (57 tok/s, ~45ms TTFT warm)
│
├─ Strongest dense model?
│  ├─ llama.cpp (throughput) ── llama-31b-rtx.yml (65 tok/s, 262K ctx)
│  └─ vLLM NVFP4 (latency) ── vllm-31b-nvfp4-rtx.yml (39 tok/s, 95ms TTFT)
│
└─ Need tool calling / structured output?
   └─ vLLM NVFP4 ──────────── vllm-31b-nvfp4-rtx.yml (Blackwell required)
```

## Performance — RTX PRO 6000 Blackwell (96 GB GDDR7)

| Compose file | Backend | Model | Quant | Speculative | tok/s | TTFT |
|---|---|---|---|---|---|---|
| `llama-e2b-rtx.yml` | llama.cpp | E2B dense (2.3B eff.) | Q8_0 | — | **276.3** | ~2.5 s |
| `llama-26b-rtx.yml` | llama.cpp | 26B MoE (3.8B active) | UD-Q4_K_XL | — | **196.2** | ~3.8 s |
| `llama-e4b-rtx.yml` | llama.cpp | E4B dense (4.5B eff.) | Q8_0 | — | **171.0** | ~1.8 s |
| `vllm-26b-nvfp4-rtx.yml` | vLLM NVFP4 | 26B MoE (3.8B active) | NVFP4 | — | **154.4** | **21 ms** |
| `vllm-12b-bf16-rtx.yml` | vLLM BF16 | 12B dense (12B) | BF16 | — | **57.3** | **127 ms** (45 ms warm) |
| `vllm-12b-fp8-rtx.yml` | vLLM FP8 | 12B dense (12B) | FP8 (3rd-party) | — | ❌ won't load | — |
| `vllm-12b-nvfp4-rtx.yml` | vLLM NVFP4 | 12B dense (12B) | NVFP4 (3rd-party) | — | ❌ won't load | — |
| `llama-31b-rtx.yml` | llama.cpp | 31B dense (30.7B) | Q4_K_M | — | 64.6 | ~9.0 s |
| `vllm-31b-fp8-rtx.yml` | vLLM FP8 | 31B dense (30.7B) | FP8-block | — | **43.1** | **89 ms** |
| `vllm-31b-nvfp4-rtx.yml` | vLLM NVFP4 | 31B dense (30.7B) | NVFP4 | — | 37.1 | 120 ms |
| `vllm-31b-fp8-mtp-rtx.yml` | vLLM FP8 | 31B dense (30.7B) | FP8-block | MTP draft=4 | 33.3 | 85 ms |
| `vllm-31b-nvfp4-mtp-rtx.yml` | vLLM NVFP4 | 31B dense (30.7B) | NVFP4 | MTP draft=4 | ❌ broken | — |
| `vllm-26b-nvfp4-mtp-rtx.yml` | vLLM NVFP4 | 26B MoE (3.8B active) | NVFP4 | MTP draft=4 | ❌ broken | — |

Benchmarked with `test_chat.py --runs 3 --warmup --no-think` on default prompt (2026-05-15 batch on `vllm/vllm-openai:gemma4-0505-cu130`; 12B row added 2026-06-04 on `vllm/vllm-openai:gemma4-unified-x86_64-cu130`).

**12B findings (2026-06-04)**:
- The 12B ships with the newer `model_type: gemma4_unified` (`Gemma4UnifiedForConditionalGeneration`), which the `gemma4-0505-cu130` image does NOT recognize ("does not recognize this architecture"). It requires the newer `vllm/vllm-openai:gemma4-unified-x86_64-cu130` image (2026-06-03). Note `google/gemma-4-31B-it` is still plain `gemma4` and stays on the 0505 image.
- **BF16 base (`google/gemma-4-12B-it`) is the only working vLLM 12B**: 57.3 tok/s, 45 ms warm TTFT, ~24 GB. As a 12B-active dense model it lands ~3× slower than the 26B MoE (3.8B active) — expected, MoE wins on decode throughput.
- **No first-party FP8/NVFP4 exists** for 12B (RedHatAI/NVIDIA only published 26B/31B). The two community quants tried both **fail to load** on the unified image with `ValueError: no module or parameter named 'embed_vision.multimodal_embedder'` — they were quantized retaining a vision-embedder weight the current `Gemma4UnifiedForConditionalGeneration` doesn't map:
  - `AxionML/Gemma-4-12B-FP8` (modelopt FP8) — ❌
  - `vrfai/gemma-4-12B-it-nvfp4` (compressed-tensors NVFP4) — ❌
- The broken FP8/NVFP4 compose files are kept (clearly marked third-party) so a future image/quant revision can be retried quickly.

**MTP findings (2026-05-15)**:
- vLLM's Gemma 4 MTP path (PR #41745) requires the `vllm/vllm-openai:gemma4-0505-cu130` image — the older `gemma4-cu130` tag does NOT have it, and the assistant model's `gemma4_assistant` arch is not in stable Transformers's `AutoModel` registry.
- **NVFP4 target + assistant drafter: AssertionError at drafter weight-load** (fused column-projection shape mismatch — see vllm issue tracker for #41789). Same bug on both 31B and 26B.
- **FP8-block + MTP runs cleanly but is ~23 % slower than FP8 vanilla** (43.1 → 33.3 tok/s). Draft acceptance against the quantized target is too low to overcome draft+verify overhead.
- BF16 target (the recipe's tested config) doesn't fit our 30 GB host RAM for either size, so we can't reproduce the recipe's benchmarks directly.
- **Stick to vanilla on Blackwell + quantized targets.** MTP is currently a regression here, not a speedup.

### Key observations

- **26B MoE is the sweet spot** — 196 tok/s rivals Qwen3.5-35B (194 tok/s) while achieving ~97% of 31B quality on benchmarks.
- **vLLM NVFP4** trades throughput for latency: 39 tok/s but 95ms TTFT vs 9s on llama.cpp. Better for interactive / agentic use.
- **E2B at 276 tok/s** is the fastest Gemma 4 variant — useful for lightweight tasks where quality is secondary.

### SGLang vs vLLM — 26B-A4B (single-stream, `test_chat.py --runs 3 --warmup`, 2026-06-29)

| Compose file | Engine | Quant | Avg tok/s | Avg TTFT | Tools |
|---|---|---|---|---|---|
| `vllm-26b-nvfp4-rtx.yml` | vLLM | NVFP4 | **178.8** | 63 ms | 0/4 † |
| `sglang-26b-nvfp4-rtx.yml` | SGLang (flashinfer_cutlass MoE) | NVFP4 | 157.1 | **24 ms** | **4/4** |
| `sglang-26b-bf16-rtx.yml` | SGLang | BF16 | ❌ won't load ‡ | — | — |

† The existing vLLM NVFP4 compose has **no `--tool-call-parser`** configured, so tool
calls aren't parsed (0/4) — a config gap, not an engine limit. The SGLang compose sets
`--tool-call-parser gemma4` and passes 4/4. (vLLM re-measured here at 178.8 tok/s on the
current image vs the 154.4 in the table above — newer image/warm-up.)

‡ **BF16 is infeasible on this host**: 30 GB system RAM vs a single ~50 GB safetensors
shard → `mmap ... Cannot allocate memory` at load, for *either* engine. This is the same
host-RAM limit that makes vLLM use NVFP4. **NVFP4 (~12 GB) is the only viable 26B path
here**, and SGLang serves it cleanly.

**Finding that overturns prior expectation:** CUTLASS NVFP4 grouped-GEMM MoE was reported
broken on SM120 (sglang #19637 / cutlass #3096 / vllm #31085). On the current
`lmsysorg/sglang:latest` with `--moe-runner-backend flashinfer_cutlass` it runs **clean**
on the RTX PRO 6000 — verified coherent output, 157 tok/s. (Still avoid the SM100-only
`trtllm_*` backends.) vLLM keeps a throughput edge (~14%); SGLang has much lower TTFT.

## Service Variants

| File | Backend | Model | GGUF / Quant | Size | Port |
|---|---|---|---|---|---|
| `llama-e2b-rtx.yml` | llama.cpp | gemma-4-E2B-it | Q8_0 | ~5 GB | 11450 |
| `llama-e4b-rtx.yml` | llama.cpp | gemma-4-E4B-it | Q8_0 | ~8 GB | 11451 |
| `llama-26b-rtx.yml` | llama.cpp | gemma-4-26B-A4B-it | UD-Q4_K_XL | ~17 GB | 11452 |
| `llama-31b-rtx.yml` | llama.cpp | gemma-4-31B-it | Q4_K_M | ~18 GB | 11453 |
| `vllm-31b-nvfp4-rtx.yml` | vLLM | Gemma-4-31B-IT-NVFP4 | NVFP4 | ~31 GB | 11453 |
| `vllm-31b-fp8-rtx.yml` | vLLM | gemma-4-31B-it-FP8-block | FP8-block | ~31 GB | 11453 |
| `vllm-31b-nvfp4-mtp-rtx.yml` | vLLM + MTP | NVFP4 + assistant drafter | NVFP4 + spec | ~31 GB | 11453 |
| `vllm-31b-fp8-mtp-rtx.yml` | vLLM + MTP | FP8-block + assistant drafter | FP8 + spec | ~31 GB | 11453 |
| `vllm-26b-nvfp4-rtx.yml` | vLLM | Gemma-4-26B-A4B-NVFP4 | NVFP4 | ~12 GB | 11454 |
| `vllm-26b-nvfp4-mtp-rtx.yml` | vLLM + MTP | NVFP4 + assistant drafter | NVFP4 + spec | ~13 GB | 11454 |
| `sglang-26b-nvfp4-rtx.yml` | SGLang | Gemma-4-26B-A4B-NVFP4 | NVFP4 | ~12 GB | 11457 |
| `sglang-26b-bf16-rtx.yml` | SGLang ✗RAM | google/gemma-4-26B-A4B-it | BF16 | ~52 GB | 11456 |
| `vllm-12b-bf16-rtx.yml` | vLLM † | google/gemma-4-12B-it | BF16 | ~24 GB | 11455 |
| `vllm-12b-fp8-rtx.yml` | vLLM † | AxionML/Gemma-4-12B-FP8 ⚠️ 3rd-party | FP8 | ~13 GB | 11455 |
| `vllm-12b-nvfp4-rtx.yml` | vLLM † | vrfai/gemma-4-12B-it-nvfp4 ⚠️ 3rd-party | NVFP4 | ~9 GB | 11455 |

† 12B configs use the `vllm/vllm-openai:gemma4-unified-x86_64-cu130` image (not `gemma4-0505-cu130`). ⚠️ The two third-party quants currently fail to load — see the 12B findings above; only the BF16 base works.

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
- **vLLM Gemma 4 support** requires the `vllm/vllm-openai:gemma4-0505-cu130` image (PR #41745 dedicated MTP path). The older `gemma4-cu130` tag works for vanilla decoding but lacks MTP; standard `vllm/vllm-openai:latest` lacks Gemma 4 entirely.
- **vLLM 26B-A4B is now viable on NVFP4** at ~12 GB on disk (~18 GB loaded). The original "BF16 won't fit RAM" note still applies to BF16 (~52 GB).
- **vLLM image is split by `model_type`**: 26B/31B (`model_type: gemma4`) run on `gemma4-0505-cu130`; the 12B (`model_type: gemma4_unified`) needs `gemma4-unified-x86_64-cu130`. Always check `config.json` `model_type` before picking the image tag.
- **Chat template** is custom (not ChatML): `<|turn>user ... <turn|>`. Embedded in GGUF metadata; no separate template file needed.
- **Recommended sampling**: temperature=1.0, top_p=0.95, top_k=64.
