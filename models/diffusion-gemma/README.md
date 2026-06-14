# DiffusionGemma 26B-A4B-it

A **discrete diffusion language model (dLLM)** on the Gemma 4 backbone, served via
vLLM with an OpenAI-compatible API. Unlike an autoregressive (AR) transformer that
emits one token at a time, DiffusionGemma **denoises a fixed 256-token canvas in
parallel** and only re-runs the tokens it isn't yet confident about (entropy-bound
denoising + self-conditioning). The payoff is decode throughput several times higher
than a same-size AR model.

- Upstream write-up: https://vllm-project.github.io/2026/06/10/diffusion-gemma.html
- vLLM recipe: https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it
- **Diffusion vs autoregressive comparison: [`comparison.html`](comparison.html)** — DiffusionGemma
  vs the same-size autoregressive Gemma 4 26B-A4B on this card (3.6–4.1× the decode tok/s, but ~27× higher TTFT).
- Base model: [`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
  — `model_type: diffusion_gemma`, arch `DiffusionGemmaForBlockDiffusion`,
  MoE with ~25.8 B total / ~3 B active (128 experts), 256-token canvas block.

## Compose files

| File | Checkpoint | Weights | Notes |
|------|-----------|---------|-------|
| `docker-compose.vllm-26b-nvfp4-rtx.yml` | [`RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4) | ~13 GB | W4A4, Blackwell-native NVFP4 path |
| `docker-compose.vllm-26b-fp8-rtx.yml` | [`RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic) | ~26 GB | W8A8, the quant the vLLM blog benchmarked |

Both are **first-party RedHatAI** checkpoints (llm-compressor / compressed-tensors),
quantized from the Google base model. All three repos are public (ungated).

Both use the dedicated **`vllm/vllm-openai:gemma-x86_64-cu130`** image (vLLM 0.24.0+,
the `gemma` tag — *not* `gemma4`). Diffusion-specific serving knobs, baked into the
composes:

- `VLLM_USE_V2_MODEL_RUNNER=1` (required for the diffusion data path)
- `--hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'`
- `--max-num-seqs 4` (keep ≤ 4 — each in-flight seq holds a 256-token canvas; higher OOMs)
- `--gpu-memory-utilization 0.85`

```bash
# NVFP4 (smaller, Blackwell-native)
docker compose -f docker-compose.vllm-26b-nvfp4-rtx.yml up -d
# FP8 (blog reference quant)
docker compose -f docker-compose.vllm-26b-fp8-rtx.yml up -d
```

API at `http://localhost:11456/v1`, served model name `diffusion-gemma-26b`.
On this Blackwell card vLLM resolves attention to **TRITON_ATTN** (Gemma 4 has
heterogeneous head dims: head_dim=256, global_head_dim=512 → FA4 unavailable).

## Benchmarks — RTX PRO 6000 Blackwell (96 GB, SM 120)

Batch-size-1 generation throughput, warm runs (first run pays cudagraph capture).
Measured with `models/shared/test_chat.py` (single 256-token canvas block) plus a
direct longer-generation / concurrency harness. Each request emits one full
256-token block, so end-to-end tok/s = total generation throughput.

| Quant | bs=1 tok/s (warm) | TTFT | 2 concurrent (agg) | 4 concurrent (agg) |
|-------|------------------:|-----:|-------------------:|-------------------:|
| **NVFP4** | ~675–850 | ~400 ms | ~925 tok/s | ~1160 tok/s |
| **FP8**   | ~720–745 | ~345 ms | ~1110 tok/s | ~1380 tok/s |

- **Single-stream is ~700 tok/s** either quant — FP8 edges NVFP4 slightly here, and
  has lower TTFT. NVFP4's win is half the weight footprint (~13 GB vs ~26 GB), not speed.
- **Diffusion batches well**: throughput keeps climbing with concurrency up to the
  `--max-num-seqs 4` cap, reaching **~1.38 k tok/s aggregate** (FP8, 4 streams).
- Steady-state across 512/1024/2048-token generations was flat (~675 tok/s NVFP4),
  i.e. the per-request denoising cost amortizes cleanly across canvas blocks.

### vs. the blog's reference cards (FP8, bs=1)

| GPU | Mem BW | FP8 bs=1 tok/s |
|-----|-------:|---------------:|
| H200 (blog) | ~4.8 TB/s | 1288 |
| H100 (blog) | ~3.35 TB/s | 1008 |
| **RTX PRO 6000 Blackwell (here)** | **~1.6 TB/s** | **~720–745** |

We land at **~73 % of H100** and **~57 % of H200** single-stream — better than the
raw memory-bandwidth ratio would predict (~0.48× H100), because the Blackwell compute
and NVFP4/FP8 tensor cores carry part of the diffusion denoising load. For a workstation
card this is **5–6× faster than a same-size autoregressive Gemma would decode** — the
whole point of the diffusion approach.

## Notes / gotchas

- The model **thinks by default** (`enable_thinking: true` in the chat template). The
  shared harness's `--no-think` sets `extra_body.enable_thinking=False`, but the
  entropy-bound sampler still produces a short reasoning preamble in practice.
- `--max-num-seqs` must stay ≤ 4 (per the model card) or the 256-token canvases OOM.
- Full-feature serving (image + tools + reasoning parsing) adds
  `--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
  --limit-mm-per-prompt '{"image":7}'` — omitted here to isolate decode speed.
- Single shared GPU: benchmarking this requires stopping the resident model
  (`qwen36-27b-fp8-dflash`) first, then restarting it afterward.
