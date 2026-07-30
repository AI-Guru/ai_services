# Laguna-S-2.1-NVFP4 (poolside) — 117.6B MoE on one RTX PRO 6000

[`poolside/Laguna-S-2.1-NVFP4`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4) —
117.6B total parameters, **8.5B activated per token**, NVFP4, 262,144-token context,
licensed **OpenMDW-1.1** (commercial use permitted).

First-party checkpoint: poolside quantize it themselves from `poolside/Laguna-S-2.1`.
Not a community quant.

Benchmark write-ups:
- [`benchmark-laguna-s-118b.html`](benchmark-laguna-s-118b.html) — single-stream speed + DFlash sweep
- [`20260730-laguna-s-concurrency-sweep.html`](20260730-laguna-s-concurrency-sweep.html) — GuideLLM concurrency/stress sweep

## Results (RTX PRO 6000 Blackwell, 96 GB, vLLM 0.26.0)

| Config | Warm tok/s | Warm TTFT | Cold TTFT | vs baseline |
|---|---|---|---|---|
| Baseline (no speculation) | **107.3** | 32 ms | 73 ms | ref |
| DFlash k=4 | 124.3 | 50–57 ms | 682 ms | +15.8% |
| DFlash k=6 | 123.0 | 50–59 ms | 693 ms | +14.6% |
| DFlash k=8 | 124.6 | 52–58 ms | 3051 ms | +16.1% |
| DFlash k=15 *(model card default)* | 108.6 | 54–62 ms | 3050 ms | +1.2% |

Single stream, batch 1, `test_chat.py --runs 5 --warmup`. Warm figures drop run 1.
Baseline and k=6 over 5 runs; k=4/8/15 over 3 runs each.

**The headline tuning result:** the model card's `num_speculative_tokens: 15` buys
essentially nothing here (+1.2%). Draft depths of 4–8 deliver ~15%. Treat k=4, k=6 and
k=8 as a tie — they span 123.0–124.6 tok/s and individual warm runs ranged 120.2–135.2,
so the sweep resolves "4–8 good, 15 bad" but not which of 4/6/8 is best.

Why: vLLM's per-position acceptance shows the drafter decaying fast — 0.66, 0.37, 0.22,
0.13, 0.11, 0.07 at k=6, and past position ~8 it is at or near zero. Mean acceptance
length only moves from 2.55 (k=6) to 2.57 (k=15), so the extra 9 drafted tokens buy 0.02
accepted tokens per step while the target model pays to verify all of them.

poolside report 2.9–3.1 mean acceptance on the DGX Spark; we measure 2.28–2.62 on this
prompt mix.

## Under load (GuideLLM concurrency ladder 1→64)

Base model, no speculation, served at `--gpu-memory-utilization 0.94` / `--max-num-seqs 64`
(KV pool 577,749 tokens). Full detail in the concurrency-sweep HTML.

| Scenario | Shape (in/out) | out:in | Peak server tok/s | Peaks at |
|---|---|---|---|---|
| chat | 2000 / 300 | 0.150 | **998.0** | > 64 (still scaling) |
| codegen | 4000 / 1500 | 0.375 | 724.9 | > 64 (still scaling) |
| rag | 8000 / 256 | 0.032 | 339.1 | 32 |
| agentic | 16000 / 800 | 0.050 | 267.8 | 16 |
| summarization | 12000 / 300 | 0.025 | 242.9 | 16 |

**Saturation is set by the output:input ratio, not request count.** Decode-heavy
workloads (codegen 0.375, chat 0.150) were still scaling at concurrency 64; prefill-heavy
ones (summarization 0.025, rag 0.032, agentic 0.050) peak at 16–32 and then go *backwards*.
Prefill competes with decode for the same SMs.

**Tail latency binds well before throughput does.** chat holds a 253 ms TTFT p95 at
concurrency 16 but 4,753 ms at 64; summarization goes 1,342 ms at 8 → 44,010 ms at 64.
Size interactive workloads off p95, not off peak tok/s.

Two non-events worth recording:

- **Over-subscribing KV queues, it does not preempt.** agentic@64 needs 1,075k tokens
  against a 577,749-token pool; `grep -c preempt` on the vLLM log returns 0. The scheduler
  refuses admission and queues instead, which is why agentic TTFT p50 jumps 1,895 ms (@16)
  → 14,346 ms (@32).
- **The 23 "errors" at agentic@64 are a harness artifact.** vLLM logged zero errors all
  sweep; guidellm hit `duration_exceeded` on its 90 s window while median latency was 72 s.

Every scenario measured 104.9–107.9 tok/s per request at concurrency 1 — an independent
reproduction of the 107.3 tok/s single-stream figure on a different harness.

Run it with [`bench-guidellm-concurrent.sh`](bench-guidellm-concurrent.sh).

## Memory

| Quantity | Value |
|---|---|
| NVFP4 weights | ~72 GB (15 shards) |
| VRAM in use | 88.7 GB at `--gpu-memory-utilization 0.90` |
| GPU KV cache | 448,413 tokens (FP8) |
| Max concurrency @ 262,144 ctx | 1.71× |
| Engine init (weights cached) | 48.3 s, incl. 22.0 s compilation |
| DFlash drafter | ~2.1 GB, shares target `embed_tokens` + `lm_head` |

The full 262K context fits with room to spare because 36 of the 48 layers use
sliding-window attention capped at a 512-token window. Only the 12 global layers scale
with context: 12 × 8 KV heads × 128 dim × 2 (K+V) × 1 byte ≈ **24 KB/token**, so 262K
tokens cost ~6.4 GB of KV.

## Usage

```bash
# baseline
docker compose --env-file ../../.env -f docker-compose.vllm-118b-nvfp4-rtx.yml up -d

# DFlash — SPEC_TOKENS sweeps the draft depth (default 4)
SPEC_TOKENS=6 docker compose --env-file ../../.env \
  -f docker-compose.vllm-118b-nvfp4-dflash-rtx.yml up -d
```

Endpoint `http://localhost:11440/v1`, served model name `laguna-s-2.1`.
Both composes use port 11440 — run one or the other, never both.

First boot downloads ~72 GB; with a warm HF cache it is ~3 min to serving.
**This model needs the whole card** — stop any other resident model first.

## Three fixes needed to boot

All three are baked into the compose files. Each presented as a generic crash-loop far
from its cause.

1. **flashinfer version skew.** Installing `flashinfer-python`/`-cubin` from PyPI stable
   while taking `-jit-cache` from the cu130 nightly index (what the other composes in
   this repo do) gives python 0.6.15 against a 0.6.16 cubin, and the NVFP4 MoE dies at
   load with `TypeError: Mismatched number of arguments when calling init(...). Expected
   9 but got 8` in `flashinfer/fused_moe/core.py:403`. Fix: pin all three to the same
   nightly version via `FI_VERSION`. Note `flashinfer-python` and `-cubin` live at
   `/whl/nightly/` but `-jit-cache` is under `/whl/nightly/cu130/`, so both
   `--extra-index-url`s are required.

2. **quack-kernels vs cutlass-dsl.** The base image's quack-kernels 0.4.1 is incompatible
   with the nvidia-cutlass-dsl 4.6.0 the vLLM upgrade pulls in →
   `AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'` during the MoE
   router warmup. That warmup is gated only on compute capability ≥ 9.0, so no serve flag
   can skip it. Fix: `pip install -U quack-kernels`. (Same fix as `models/soofi-s/`.)

3. **`--moe-backend triton` does not apply to NVFP4.** The vLLM recipe lists it for
   DFlash, but it targets the BF16/INT4 checkpoints. On NVFP4 vLLM rejects it:
   `ValueError: moe_backend='triton' is not supported for NvFP4 MoE`. Left on `auto`,
   which selects `FLASHINFER_CUTLASS` — the same backend as the baseline, which keeps the
   two configs comparable.

Also note the model card's `CUTE_DSL_ARCH=sm_121a` is for the DGX Spark's GB10. The
composes here set `sm_120a` for the RTX PRO 6000 Blackwell.

## Thinking / reasoning caveats

The chat template variable is **`enable_thinking`** (not `thinking`), and it defaults to
**true**. It must be sent as `chat_template_kwargs`, not top-level:

```bash
curl http://localhost:11440/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "laguna-s-2.1",
  "messages": [{"role":"user","content":"What is 17*23?"}],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Two traps:

- **`models/shared/test_chat.py --no-think` is a no-op here.** It sends a top-level
  `enable_thinking`, which vLLM 0.26 ignores. Every number above was therefore measured
  with thinking **on**. That does not distort tok/s, but they are not answer-only latencies.
  (Same no-op as the Soofi-S family.)
- **`reasoning_content` is never populated.** With no `chat_template_kwargs` the
  `poolside_v1` parser falls back to an identity parser and `</think>` leaks into
  `content`. With `enable_thinking: true` the reasoning is correctly stripped from
  `content` — but does not appear in `reasoning_content` either, so those tokens are
  billed and discarded. Use `enable_thinking: false` for clean direct answers.

Tool calling via `--tool-call-parser poolside_v1` works correctly (verified: emits proper
`tool_calls` with `finish_reason: tool_calls`).

## Not measured

Concurrent/batched throughput, long-context prefill rate, and quality. poolside report
70.2% Terminal-Bench 2.1, 78.5% SWE-bench Multilingual, 59.4% SWE-Bench Pro — none
independently verified here.
