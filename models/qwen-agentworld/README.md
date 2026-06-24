# Qwen-AgentWorld-35B-A3B

[Qwen-AgentWorld-35B-A3B](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B) is Qwen's
agentic **world model** — instead of being the agent, it *simulates the environment*, predicting
state transitions in response to an agent's actions across seven domains: **MCP** (tool calling),
**Search**, **Terminal**, **SWE**, **Android**, **Web**, and **OS**. Domain-specific system prompts
live in the upstream GitHub repo's `prompts/` directory.

It is a fine-tune of **Qwen3.5-35B-A3B-Base**, so it inherits that exact architecture — and the same
deployment caveats as the [`qwen3.5/`](../qwen3.5/) family.

## Architecture

- **Hybrid 40 layers**: 30 Gated DeltaNet (linear-attention) + 10 full Gated-Attention, in the
  pattern `10 × (3 × (DeltaNet → MoE) → 1 × (Attention → MoE))`.
- **MoE**: 256 experts, 8 routed + 1 shared active per token → **35B total / 3B active**.
- **Context**: 262,144 tokens (keep ≥128K — the world model relies on long context for multi-turn
  environment simulation).
- **Text-only.** Thinking-on by default (`<think>…</think>`) to reason about state transitions.
- **Recommended sampling**: `temperature=0.6, top_p=0.95, top_k=20`, output ≤32,768 tokens.

Only the 10 full-attention layers carry a traditional KV cache (~20 KB/token bf16); the 30 DeltaNet
layers use a sequence-length-agnostic recurrent state (~15 MB/request). This keeps KV tiny, so the
full 262K context fits on a single 96 GB card even at **BF16** weights.

## Quick start

```bash
# First-party BF16 (vetted, ~70 GB) — vLLM
docker compose -f docker-compose.vllm-35b-bf16-rtx.yml up -d
# …or SGLang
docker compose -f docker-compose.sglang-35b-bf16.yaml up -d

# API (OpenAI-compatible)
curl http://localhost:11450/v1/models
```

All four variants serve on **port 11450** as model name **`agentworld-35b`**. They share the
`agentworld_huggingface_cache` volume, and only one runs at a time (single shared GPU).

## Compose variants

| File | Engine | Quant | Checkpoint | Notes |
|------|--------|-------|------------|-------|
| `docker-compose.vllm-35b-bf16-rtx.yml`  | vLLM   | BF16 | `Qwen/Qwen-AgentWorld-35B-A3B` | First-party, vetted. Primary. vLLM `v0.19.0-cu130` + `--language-model-only`. |
| `docker-compose.sglang-35b-bf16.yaml`   | SGLang | BF16 | `Qwen/Qwen-AgentWorld-35B-A3B` | First-party, vetted. `mem-fraction-static 0.90`. |
| `docker-compose.vllm-35b-fp8-rtx.yml`   | vLLM   | FP8  | `lovedheart/Qwen-AgentWorld-35B-A3B-FP8` | ⚠️ **Community** quant (no official Qwen FP8). Loads & runs. |
| `docker-compose.sglang-35b-fp8.yaml`    | SGLang | FP8  | `lovedheart/Qwen-AgentWorld-35B-A3B-FP8` | ⚠️ **Community** quant. Loads & runs. |

## Benchmarks (RTX PRO 6000 Blackwell, 96 GB)

`test_chat.py`, default MoE-vs-dense prompt, `--runs 3 --warmup`, single stream. tok/s and TTFT are
averaged over the 3 warm runs. Measured 2026-06-24 (vLLM `v0.19.0-cu130`, SGLang `latest`).

| Engine | Quant | Mode        | tok/s | TTFT avg |
|--------|-------|-------------|-------|----------|
| vLLM   | BF16  | no-think    | 168.3 | 8.2 s |
| vLLM   | BF16  | thinking-on | 169.0 | 8.0 s |
| vLLM   | FP8   | no-think    | 170.1 | 8.1 s |
| vLLM   | FP8   | thinking-on | 170.7 | 7.9 s |
| SGLang | BF16  | no-think    | 162.9 | 7.7 s |
| SGLang | BF16  | thinking-on | 163.5 | 8.0 s |
| SGLang | FP8   | no-think    | 162.9 | 8.2 s |
| SGLang | FP8   | thinking-on | 163.5 | 8.1 s |

**Reading the numbers:**
- **vLLM ≈ 169–171 tok/s, SGLang ≈ 163 tok/s** for single-stream decode — vLLM is ~4% faster here.
- **FP8 ≈ BF16 on throughput.** With only ~3B active params, single-stream decode isn't
  weight-bandwidth-bound, so FP8 barely moves tok/s. FP8's real benefit is the ~35 GB smaller
  footprint → far more KV headroom for concurrency / long context, not single-stream speed.
- **TTFT (~8 s) is dominated by reasoning, not latency.** AgentWorld **always thinks** (see below):
  `test_chat.py` reports TTFT as time-to-first-*answer* token, which lands only after a ~5–6k-char
  `<think>` block. It scales with reasoning length, so treat it as "think time," not request latency.
- **`--no-think` has no effect on AgentWorld** — `enable_thinking=false` is ignored and the model
  reasons anyway, so the two modes are statistically identical. Kept in the table for transparency.

## Known issues

- **vLLM needs `--language-model-only` + the v0.19.0-cu130 image.** AgentWorld's `config.json`
  declares the multimodal class `Qwen3_5MoeForConditionalGeneration` but ships **no vision weights**
  (it's text-only). Without `--language-model-only`, vLLM tries to build the vision tower and dies
  with `ValueError: Following weights were not initialized from checkpoint: {'visual.blocks...'}`.
  The flag makes vLLM run "in text-only mode"; it requires v0.19.0-cu130 (the v0.17.0 image used by
  the `qwen3.5/` family doesn't expose it). **SGLang** loads the checkpoint fine with no extra flag.
- **AgentWorld always reasons.** `enable_thinking=false` (what `--no-think` sends) is ignored — every
  response carries a `<think>` block. Expected for a world model that reasons about state transitions;
  it just means `--no-think` benchmarks ≈ thinking-on benchmarks.
- **FP8 KV cache silently corrupts output** on the DeltaNet layers (no crash, just wrong answers) —
  `vllm-project/vllm#26646`. KV is left at auto/bf16 in vLLM and pinned to `bf16` in SGLang
  (`--kv-cache-dtype bf16 --mamba-ssm-dtype bfloat16`). Do not enable FP8 KV.
- **Community FP8 quant.** There is no official Qwen FP8 of AgentWorld; the FP8 files use a
  third-party quant (`lovedheart/...`). It **did load and benchmark cleanly** on both engines here
  (2026-06-24), but it is unvetted — prefer the first-party BF16 files if correctness matters.
- **SGLang `HiRadixCache only supports MHA and MLA`** warning at startup is expected (DeltaNet blocks
  higher-order cache) and non-fatal.
- **No speculative/MTP by default.** AgentWorld is a fine-tune; we don't assume its NEXTN draft heads
  survived. The SGLang BF16 header documents the flags to opt back in.

## Testing

```bash
# no-think (repo benchmark convention)
python3 ../shared/test_chat.py --base-url http://localhost:11450/v1 \
    --model agentworld-35b --runs 3 --warmup --no-think

# thinking-on (AgentWorld's default mode)
python3 ../shared/test_chat.py --base-url http://localhost:11450/v1 \
    --model agentworld-35b --runs 3 --warmup
```

Also available in [`../shared/`](../shared/): `test_tools.py` (tool-call integration),
`test_scenarios.py` (Chat/RAG/CodeGen/Summarization/Agentic throughput).

## Requirements

- NVIDIA GPU with FP8/BF16 support; benchmarked on RTX PRO 6000 Blackwell (96 GB, SM 120).
- Docker + NVIDIA Container Toolkit.
- All checkpoints are public — no `HF_TOKEN` needed (a gitignored `.env` can still supply one).
