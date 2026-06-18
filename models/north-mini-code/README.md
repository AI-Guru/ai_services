# North-Mini-Code Self-Hosted Inference

[CohereLabs/North-Mini-Code-1.0](https://huggingface.co/CohereLabs/North-Mini-Code-1.0)
— a **30B-total / 3B-active** open-weight Mixture-of-Experts model from Cohere Labs,
optimized for **code generation, agentic software engineering, and terminal tasks**.
Apache 2.0. Released June 2026.

Served here via vLLM (FP8, primary) on an OpenAI-compatible API, with a llama.cpp
GGUF fallback.

## Architecture

- **Sparse MoE**: 128 experts, **8 active per token** (3B active of 30B total), single
  dense layer before the sparse stack, sigmoid-gated router, SwiGLU experts.
- **Attention**: custom 3:1 ratio of sliding-window (RoPE, window 4096) to global attention.
- **`model_type: cohere2_moe`** (`Cohere2MoeForCausalLM`): 49 layers, hidden 2048,
  32 heads / 4 KV heads, head_dim 128, vocab 262144.
- **Context**: 256K input / 64K max output. **License**: Apache 2.0. Weights: BF16.
- Post-trained via cascaded SFT + RLVR; supports tool-use and interleaved thinking.
- Recommended sampling: `temperature=1.0`, `top_p=0.95`.

## Requirements

- NVIDIA GPU with ~32+ GB VRAM for FP8 (RTX PRO 6000 96 GB here); Docker + NVIDIA Container Toolkit.
- **vLLM**: needs the **main branch** (`vllm/vllm-openai:nightly`, built from main) — no
  tagged release registers `cohere2_moe` yet. The compose's boot step installs, in order,
  on top of the nightly image (no custom image required, ~minutes on first start):
  1. latest vLLM nightly wheel;
  2. **`transformers` from git main** — the nightly's bundled transformers is too old and
     errors `model type cohere2_moe ... does not recognize this architecture` (needs `git`,
     which the boot step `apt-get install`s since the image lacks it);
  3. **`cohere_melody>=0.9.0`** for the `cohere_command4` tool-call / reasoning parser.
  Also sets `FLASHINFER_DISABLE_VERSION_CHECK=1` — the nightly pulls flashinfer 0.6.12
  while the image's jit-cache is 0.6.8; the versions are compatible, the equality guard isn't.
- **llama.cpp**: stock `ghcr.io/ggml-org/llama.cpp:server-cuda` (no melody needed).
- `HF_TOKEN` in project-root `.env`.

## Port

| Port | Model |
|------|-------|
| 11460 | North-Mini-Code (vLLM **and** llama.cpp — one port per model, never both up at once) |

## Compose variants

| File | Engine | Quant | Checkpoint | Notes |
|------|--------|-------|-----------|-------|
| `docker-compose.vllm-30b-fp8-rtx.yml` | vLLM | FP8 | `CohereLabs/North-Mini-Code-1.0-fp8` (first-party) | **Primary, benchmarked.** nightly image + boot-installs transformers-main / cohere_melody. 262K ctx, tp=1. |
| `docker-compose.llama-30b-q4-rtx.yml` | llama.cpp | UD-Q4_K_XL | `unsloth/North-Mini-Code-1.0-GGUF` (community, unvetted) | Fallback — no melody / vLLM-main dependency. **Not benchmarked** (vLLM path succeeded); quant tag unverified. |

## Benchmarks (RTX PRO 6000, 96 GB)

Tested 2026-06-18 with `test_chat.py --runs 3 --warmup --no-think` on the default MoE
prompt, `vllm/vllm-openai:nightly` (vLLM 0.23.1rc1.dev149, transformers 5.13.0.dev0,
cohere_melody 0.9.0). tok/s counts all generated tokens over full wall time.

| Run | TTFT | tokens | tok/s |
|-----|-----:|-------:|------:|
| 1 | 351 ms | 1988 | 177.0 |
| 2 | 239 ms | 1427 | 183.5 |
| 3 | 274 ms | 1972 | 177.3 |
| **avg** | **288 ms** | — | **179.3** |

Throughput is tight (177–184 tok/s) and TTFT is very low (~0.24–0.35 s warm) — the FP8
30B-A3B is decode-light (only 3B active) and prefill on the short prompt is trivial.
North-Mini-Code has **no published MTP head / drafter**, so this is a no-speculative-decoding
baseline; the figure is purely the FP8 MoE decode rate. (`--no-think` is only partly
honored — the model still emits a brief `<think>` block, ~250–340 chars, before answering.)

**Tool calling**: `test_tools.py` → **4/4 passed** (single, parallel-2, chained,
multi-parallel-3) via the `cohere_command4` parser — emits proper structured
`tool_calls`, including genuine parallel calls.

## tok/s comparison — same-size models in this repo

All four are **~30B-class, ~3B-active MoE** — the closest form-factor matches we have.
Single-user `test_chat.py` decode throughput on the same RTX PRO 6000 Blackwell.
Engine / quant / speculative-decoding noted per row since they differ (apples to
oranges otherwise).

| Model | Total / active | Engine · quant · spec | tok/s |
|-------|----------------|-----------------------|------:|
| **North-Mini-Code-1.0** | 30B / 3B MoE | **vLLM · FP8 · no-spec** | **179.3** |
| Qwen3.6-35B-A3B | 35B / 3B MoE | vLLM · FP8 text-only · no-spec | 203.2 |
| Qwen3.6-35B-A3B | 35B / 3B MoE | vLLM · NVFP4 · MTP n=3 | 367.7 |
| Nemotron-Cascade-2-30B-A3B | 30B / 3B MoE | vLLM · AWQ-int4 · no-spec | 351 |
| Nemotron-3-Nano-30B-A3B | 30B / 3B MoE | vLLM · FP8 · no-spec | ~296 |
| Gemma4 26B-A4B | 25.2B / 3.8B MoE | vLLM · NVFP4 · no-spec | 154.4 |
| Gemma4 26B-A4B | 25.2B / 3.8B MoE | llama.cpp · Q4_K_XL · no-spec | 196.2 |

Comparison numbers sourced from each family's README (`../qwen3.6/`, `../nemotron/`,
`../gemma4/`).

**Takeaway:** at **179 tok/s** on the closest like-for-like config (vLLM FP8, no
speculative decoding), North-Mini-Code lands right where a 3B-active MoE should — a hair
behind Qwen3.6-35B-A3B's FP8 text-only (203) and ahead of Gemma4 26B-A4B's vLLM NVFP4
(154). The much higher numbers in the table come from advantages North-Mini-Code can't
yet match on this box: speculative decoding (Qwen NVFP4+MTP at 368) or a lighter INT4
quant (Nemotron Cascade-2 AWQ-int4 at 351); Nemotron Nano's ~296 is FP8-vs-FP8 but on a
Mamba-2 hybrid whose decode is cheaper than this dense-attention MoE. North-Mini-Code
ships no MTP/drafter, so FP8 decode is the ceiling today — a fair-fight FP8 comparison
puts it squarely mid-pack, with the bonus that its `cohere_command4` tool calling is
clean 4/4.

## Testing

```bash
# Chat benchmark — measures TTFT and tok/s
python3 ../shared/test_chat.py --base-url http://localhost:11460/v1 --model north-mini-code --runs 3 --warmup --no-think

# Tool-calling integration test (cohere_command4 parser)
python3 ../shared/test_tools.py --base-url http://localhost:11460/v1 --model north-mini-code
```
