# Mistral Medium 3.5 (128B)

Mistral AI's flagship open-weights model from 2026-04-29 — dense 128B, multimodal (text + vision), 256K context, modified-MIT license. Designed for long-horizon coding, agentic, and reasoning tasks.

[Mistral blog announcement](https://mistral.ai/news/vibe-remote-agents-mistral-medium-3-5) · [Native weights (gated)](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) · [GGUF quants (public)](https://huggingface.co/bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF)

## Why we run this

The 128B dense architecture beats most open MoE alternatives on coding-quality benchmarks (SWE-Bench Verified 77.6, τ³-Telecom 91.4). Its 256K context and native tool-calling make it a serious option for single-user agentic workloads. The trade-off vs the MoE 35B-A3B running on the same GPU is decode speed — the 35B is ~12× faster but sometimes weaker per-token.

## Architecture

- **Dense**, 128B parameters (all active per token)
- ~88 layers, 8 KV heads, head_dim 128 (Mistral 3 family standard)
- **Multimodal** vision encoder trained from scratch — disabled here via `--no-mmproj`
- **256K context** native (we configure 64K to keep KV in-budget; raise as needed)
- Tekken tokenizer, multilingual (en, fr, de, es, pt, it, ja, ko, ru, zh, ar, fa)
- Modified MIT license (open weights, see `LICENSE` in upstream repo for details)

## Why this fits on a single RTX PRO 6000 (96 GB)

| Quant | Weights | KV headroom | Note |
|---|---:|---:|---|
| Q8_0 | 132.9 GB | doesn't fit | — |
| Q5_K_M | 91.1 GB | ~5 GB | only short context |
| **Q4_K_M** | **78.4 GB** | **~13 GB** | **default — chosen here** |
| Q4_K_S | 73.0 GB | ~18 GB | lighter, similar quality |
| IQ4_XS | 69.1 GB | ~22 GB | most KV headroom |
| Q3_K_M | 63.3 GB | ~28 GB | quality drop noticeable |

Q4_K_M is the community standard for 128B-class dense models. With q8_0 KV-cache quantisation we estimate ~64–80 K usable context (Mistral 3: 88 layers × 8 KV × 128 head_dim ≈ 191 KB/token at q8 KV). We configure `-c 65536` as a safe default; bump to `131072` if you keep generations short and your prompts don't fill the cache.

## Performance expectations

- **Theoretical ceiling**: 1600 GB/s ÷ 78 GB weights ≈ **20 tok/s decode**
- **Realistic**: 12–18 tok/s on the RTX PRO 6000
- **TTFT**: depends on prompt size; expect 1–5 s for short prompts, much longer for long contexts (chunked prefill)

For comparison on the same GPU:
- Qwen3.6-35B-A3B FP8: ~200 tok/s (MoE, 3B active)
- Qwen3.6-27B FP8 + DFlash: ~100 tok/s (dense + spec decoding)
- **Mistral Medium 3.5 Q4_K_M: ~15 tok/s** (dense, no spec)

The 128B is the right pick when per-token quality matters more than throughput.

## Quick start

```bash
docker compose -f docker-compose.llama-128b-rtx.yml up -d
docker logs -f mistral-medium-128b   # watch the GGUF download (~78 GB, 10-15 min)
```

API endpoint: `http://localhost:11470/v1`, model name: `mistral-medium-3.5`

```bash
curl http://localhost:11470/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-medium-3.5",
    "messages": [{"role": "user", "content": "Write a haiku about dense transformers."}],
    "max_tokens": 200
  }'
```

## Compose variants

| File | Engine | Quant | Notes |
|------|--------|-------|-------|
| `docker-compose.llama-128b-rtx.yml` | llama.cpp | Q4_K_M | Default. Single-user, 64K context, vision off. |

vLLM/SGLang aren't included by default. Community feedback (April 2026) describes Blackwell (SM_120) support for vLLM with this model size as still maturing — frequent build/kernel issues. llama.cpp is the reliable path for single-user use.

## Known limitations

- **Slow vs MoE**: ~15 tok/s vs 100–200 for the Qwen3.6 family on the same GPU. Expected — the 128B has 4× more active params per token.
- **No vision**: `--no-mmproj` disables the bf16 vision projector to save ~6 GB VRAM. Add a multimodal compose variant if you need image input.
- **Tool calling format differs from Qwen**: Mistral 3 uses its own structured tool-call format. Use `--jinja` (which we do) so the embedded chat template handles it; client-side parsing may be required for some agent frameworks expecting the OpenAI-style structured `tool_calls` array.
- **Q4 quality loss**: ~0.5–2 % vs native precision on coding benchmarks. Acceptable for daily use; not a substitute for the API for evaluation work.
- **30 GB host RAM**: 78 GB GGUF can be tight to mmap during initial load. `--mlock` keeps weights pinned in GPU memory after load to avoid swap thrashing during long idles.

## Related

- [`models/qwen3.6/`](../qwen3.6/) — fast MoE/dense alternatives at 27B–35B
- [`models/qwen3.5/`](../qwen3.5/) — the previous-generation 122B dense option (heavier weights, same use case)
