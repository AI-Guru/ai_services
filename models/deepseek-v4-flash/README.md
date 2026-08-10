# DeepSeek-V4-Flash

[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) — 284B-total / 13B-active Mixture-of-Experts model from DeepSeek (MIT license).

The compact sibling of DeepSeek-V4-Pro (1.6T). Despite 13B active parameters it beats the V4-Pro *preview* on every agentic benchmark DeepSeek published.

## Architecture

- **Hybrid sparse attention**: Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA), driven by a 64-head Lightning Indexer (`index_topk=512`)
- **MoE**: 256 routed experts, 6 routed + 1 shared active per token, 43 layers
- **mHC**: manifold-constrained hyper-connections (TileLang kernels)
- **Mixed-precision weights**: MoE experts natively **FP4 (MXFP4)**, attention / norms / router **FP8** block-wise (128×128, `ue8m0` scales)
- **Context**: 1,048,576 native — no YaRN override needed
- **Efficiency**: 27% of V3.2's per-token inference FLOPs, ~10% of its KV cache at 1M
- **Speculative decoding**: DSpark draft folded into the checkpoint (`dspark_block_size=5`) — *not usable on SM120*, see caveats
- **License**: MIT

## Quick start

```bash
# 2x RTX PRO 6000 (2x96 GB) — requires both cards
docker compose -f docker-compose.vllm-284b-mxfp4-256k-rtx.yml build   # ~1 min, FlashInfer bump
docker compose -f docker-compose.vllm-284b-mxfp4-256k-rtx.yml up -d   # recommended

# Full 1M context instead (lower throughput under load):
# docker compose -f docker-compose.vllm-284b-mxfp4-1m-rtx.yml up -d
```

API endpoint: `http://localhost:11437/v1`, model name: `deepseek-v4-flash`. Cold start ~4 min.

Reasoning effort is controlled through `chat_template_kwargs`, **not** `enable_thinking` (that is the Qwen convention and this model ignores it):

```python
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
)
```

Recommended sampling: `temperature=1.0`, `top_p=0.95` for agentic work (`1.0` otherwise).

## Compose variants

| File | Engine | Quant | Context | Notes |
|------|--------|-------|--------:|-------|
| `docker-compose.vllm-284b-mxfp4-256k-rtx.yml` | vLLM 0.25.0 | MXFP4 experts + FP8 | 262,144 | **Recommended default.** util 0.94, batch budget 16384. Wins every concurrency scenario — see the sweep below |
| `docker-compose.vllm-284b-mxfp4-1m-rtx.yml` | vLLM 0.25.0 | MXFP4 experts + FP8 | 1,048,576 | Full native context. util 0.94 (forced), batch budget 8192 (forced). 15–48% less throughput under concurrency |

Both are 2× RTX PRO 6000, TP2 + expert parallelism, serve as `deepseek-v4-flash` on port
11437, and are **mutually exclusive** — each claims all GPUs and the same port.

Both require the custom image built from `Dockerfile.vllm-sm120` — stock
`vllm/vllm-openai:v0.25.0` pins FlashInfer 0.6.13, but the DeepSeek-V4 sparse-MLA decode
path calls `swa_topk_lens`, added in 0.6.14.

### Why 1M forces its settings

The KV pool must hold one full-length request. At 1M that pins utilization near 0.94,
leaving ~794 MiB free per card — no room for a larger prefill activation buffer, hence
the 8192 batch budget. The 256K variant is *not* cheaper: because per-token KV cost falls
as the declared max length grows, 256K needs **more** KV (7.55 GiB) than 1M (6.85 GiB).
It runs at the same 0.94 utilization and spends its surplus on the doubled batch budget.
An earlier attempt at 0.92 for 256K failed outright — 7.08 GiB available against 7.55
needed, "estimated maximum model length is 139,520".

## Benchmarks (2× RTX PRO 6000 Blackwell, TP2, PCIe)

Measured 2026-08-10, vLLM 0.25.0 + FlashInfer 0.6.14, `test_chat.py` / `test_scenarios.py` (3 runs + warmup), non-thinking mode.

### Throughput

| Config | Context | tok/s | TTFT |
|--------|--------:|------:|-----:|
| **MXFP4 TP2, 32K ctx** | 32,768 | **97.9** | 48 ms |
| **MXFP4 TP2, 1M ctx** | 1,048,576 | **95.1** | 48 ms |

Going from 32K to the full 1M context costs **2.9%** throughput. The context length is essentially free — what it costs is concurrency.

### Scenario sweep (32K config)

Cold first runs are excluded: RAG, summarization and agentic each take a one-time ~5.2 s TTFT hit on run 1 (first-prefill JIT/autotune for a new input-length bucket), then settle at ~70 ms.

| Scenario | In / Out | Warm tok/s | Warm TTFT |
|----------|---------:|-----------:|----------:|
| Chat | 26 / 157 | 96.0 | 44 ms |
| RAG | 732 / 300 | 96.0 | 75 ms |
| Codegen | 122 / 1000 | 97.7 | 60 ms |
| Summarization | 529 / 200 | 96.5 | 42 ms |
| Agentic | 898 / 600 | 97.1 | 68 ms |
| **Overall** | | **~96.6** | |

Throughput is essentially flat from 26 to 898 input tokens — the sparse attention keeps long context nearly free in decode.

### Against the family baselines

| Model | Backend | tok/s |
|-------|---------|------:|
| Qwen3.5-35B-A3B | vLLM FP8 | 146 |
| **DeepSeek-V4-Flash (284B)** | **vLLM MXFP4, TP2** | **~96** |
| Nemotron Super-120B | vLLM NVFP4 | 80 |

A 284B model outrunning the 120B by ~20%, with 13B active parameters doing the work.

## Context length vs concurrency

All values below are **measured**, not extrapolated — vLLM reports them at startup.
Each row is a different `--max-model-len`, so the KV pool differs per row.

| max-model-len | util | batch budget | KV memory | KV pool (tokens) | KV per token | Concurrency at full length |
|--------------:|-----:|-------------:|----------:|-----------------:|-------------:|---------------------------:|
| 32,768 | 0.90 | default | 6.85 GiB | 74,815 | 96.0 KiB | 2.28× |
| 262,144 | 0.94 | 16384 | 8.98 GiB | 350,255 | 30.2 KiB | 1.34× |
| 1,048,576 | 0.94 | 8192 | 10.77 GiB | 1,650,709 | 6.8 KiB | 1.57× |

Note the ordering: the **1M config has the largest KV pool**, not the smallest. Per-token
KV cost falls ~14× from 32K to 1M, which more than offsets the longer requests. Concurrency
at *full* declared length is low everywhere, but at realistic prompt sizes the 1M pool holds
far more requests than 256K — it simply cannot feed them, because its prefill batch budget
is half. See the sweep below.

**Do not extrapolate per-token KV cost from a short-context measurement.** At a 32K limit the pool reports 74,815 tokens for 6.85 GiB (~96 KiB/token); at 1M it reports 1,650,709 tokens for 10.77 GiB (~6.8 KiB/token) — a ~14× difference. DSv4's `compress_ratios` schedule compresses most of its 43 layers by 4× or 128×, so the effective per-token cost falls sharply as context grows. Linear extrapolation from 32K wrongly suggests 1M needs ~96 GiB of KV and a 4-GPU box. It does not.

## Concurrency scaling — GuideLLM sweep (2026-08-10)

`bench-guidellm-parallel.sh`, guidellm 0.5.4, `--profile sweep`, 120 s per rate point,
10 points per scenario, both configs on identical hardware with no other GPU load.
"1 user" is the synchronous point; "saturated" is the best of the throughput and
constant-rate points. Scaling = saturated / 1 user.

| Scenario | In / Out | 256K 1 user | 256K saturated | 256K scaling | 1M saturated | 1M scaling | Δ peak |
|----------|---------:|------------:|---------------:|-------------:|-------------:|-----------:|-------:|
| **Chat** | 2K / 300 | 76.4 | **531.0** | **7.0×** | 449.4 | 6.0× | **−15%** |
| RAG | 8K / 256 | 76.7 | 113.9 | 1.5× | 58.8 | 0.9× | **−48%** |
| Summarization | 12K / 300 | 70.0 | 108.3 | 1.5× | 66.3 | 1.0× | **−39%** |
| Agentic | 16K / 800 | 80.4 | 89.4 | 1.1× | 80.7 | 1.0× | −10% |
| Codegen | 4K / 1500 | 95.0 | 99.6 | 1.0× | 94.2 | 1.0× | −5% |

**Only chat parallelizes.** Everything else saturates at roughly single-user throughput.
Aggregate output sits at a hard ~90–115 tok/s ceiling for every workload with a prompt
above ~4K or an output above ~800 tokens. Chat is the sole exception because its small
KV footprint and short outputs let vLLM batch enough requests to break past it.

**256K beats 1M on every scenario.** The 1M config's KV pool is 4.7× larger
(1,650,709 vs 350,255 tokens) and buys nothing — it cannot use the capacity because
its prefill batch budget is half (8192 vs 16384). RAG is the extreme case: at 1M it
scales at **0.9×**, i.e. concurrent load makes it *slower* than serving one request at
a time, because an 8K prompt consumes essentially the whole 8192-token prefill step and
requests serialize with scheduling overhead on top.

Single-user throughput is within ~5% across both configs, so declaring 1M costs almost
nothing at low load — it costs throughput only under concurrency.

### Sweet spots (highest rate holding inter-token latency < 15 ms)

| Scenario | Config | req/s | Aggregate tok/s | TTFT p50 | ITL p50 | Latency p50 |
|----------|--------|------:|----------------:|---------:|--------:|------------:|
| Chat | 256K | 0.48 | 145.4 | 248 ms | 13.5 ms | 4.7 s |
| Chat | 1M | 0.44 | 131.0 | 251 ms | 13.0 ms | 4.4 s |
| RAG | 256K | 0.37 | 97.8 | 776 ms | 13.8 ms | 4.6 s |
| RAG | 1M | 0.21 | 58.8 | 807 ms | 10.4 ms | 3.7 s |
| Summarization | 256K | 0.30 | 91.8 | 1141 ms | 14.7 ms | 5.9 s |
| Agentic | 256K | 0.10 | 89.4 | 1521 ms | 12.1 ms | 12.5 s |
| Codegen | 256K | 0.06 | 99.6 | 419 ms | 10.3 ms | 16.3 s |

Codegen and agentic latency exceeds 5 s at *any* rate — that is inherent to 1500- and
800-token outputs at ~10 ms per token, not a saturation effect.

### Practical capacity

Assuming a chat user issues ~2 requests/min (0.033 req/s) with think time between turns:

| Workload | Concurrent users at 256K |
|----------|-------------------------:|
| Chat, interactive (ITL < 15 ms) | **~14** |
| Chat, tolerating ITL ~67 ms | ~60 |
| RAG | ~11 |
| Summarization | ~9 |
| Agentic (continuous, not bursty) | **~1** |

**Recommendation: serve 256K.** It wins on every measured workload, and nothing in this
set uses prompts beyond 256K. Reserve the 1M variant for single-user long-context work,
where it is within noise of 256K and slightly more responsive under load.

Raw per-rate data: `benchmarks/guidellm/{256k,1m}/<scenario>/benchmark.{json,csv}`.

### Measurement notes

- The 1M RAG synchronous point read 63.4 tok/s in the sweep; a 60 s re-run gave
  **72.9 tok/s**. A single 120 s synchronous sample carries real variance. The 0.9×
  scaling figure above uses the sweep's own baseline (58.8 / 63.4) so the table stays
  internally consistent; against the re-measured 72.9 baseline it is 0.81×. Either way
  it is below 1.0 — the saturated collapse reproduces and is the robust finding.
- Prefix caching is enabled in both configs, matching deployment. Synthetic prompts
  share little, but these are not cache-cold numbers.
- No speculative decoding (SM120 limitation), so every figure here is a floor.

## Caveats

**Speculative decoding does not work on SM120.** The checkpoint ships the DSpark draft (`dspark_block_size=5`), and DSpark/MTP are worth +40–50% on supported hardware, but both abort during warmup with `Check failed: num_tokens > 64` from `sparse_mla_sm120_paged_attention` — the SM120 sparse MLA kernel requires more than 64 tokens per batch while draft verification submits only the draft length. Unlocking it needs [vllm PR #41834](https://github.com/vllm-project/vllm/pull/41834), still unmerged as of 2026-08-10. That branch is validated first-party on a 2× RTX PRO 6000 box and would put this config around 135–145 tok/s.

**`--gpu-memory-utilization 0.97` fails.** KV allocates fine (13.2 GiB → 2,023,388 tokens) but CUDA graph capture then OOMs with 142 MiB free against a 228 MiB request. 0.94 is verified; treat it as the ceiling.

**NVFP4 is not worth it here.** [nvidia/DeepSeek-V4-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4) re-quantizes only the MoE experts to NVFP4 and is a re-quant of the **preview** checkpoint, not 0731 — that means Terminal Bench 2.1 61.8 instead of 82.7 and DeepSWE 7.3 instead of 54.4. It is also ~1.4 GB *larger* on disk (the native experts are already 4-bit MXFP4), needs PR #41834 to serve on SM120 at all, and requires expert parallelism off. The PR authors measured it on this exact hardware and concluded MXFP4 remains the better choice on consumer Blackwell.

**"Think Max" reasoning is untested here.** It requires `--max-model-len >= 393216`, which the 1M config satisfies, but no accuracy or latency run has been done at that effort level. `low` / `high` are exercised.

**No NVLink.** The two cards are PCIe-linked (`nvidia-smi topo -m` reports `NODE`), so TP2 all-reduces at every one of the 43 layers cross the PCIe root complex. The official 8× RTX PRO 6000 recipe profile is also PCIe-only, so this is the expected topology rather than a misconfiguration.

**Host RAM is 62 GB against a 155 GiB checkpoint.** vLLM detects this and disables weight auto-prefetch, streaming shards via mmap instead. Load takes ~40 s and is not a problem, but do not run other memory-hungry work during startup.

## Not yet done

- Re-sweep 256K with `--max-num-batched-tokens 24576+` — the sweep shows prefill batch
  budget is the binding constraint, so this is the single most promising lever left.
  Needs a memory check first: 0.94 utilization currently leaves ~5.8 GB free per card
  at 256K.
- Tool-calling validation via `test_tools.py` with the `deepseek_v4` parser
- Accuracy run (GSM8K / GPQA) to compare against the 95.0% / 87.4% figures reported for this hardware in vllm PR #43477
- Think Max (`reasoning_effort: max`) latency and quality at 384K+

## References

- [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) — official profiles (note: the only RTX PRO 6000 profile is 8×, TP8; the 2× TP2 config here is not upstream)
- [vllm PR #43477](https://github.com/vllm-project/vllm/pull/43477) — merged SM120 enablement, with TP=2 RTX PRO 6000 benchmarks
- [vllm PR #41834](https://github.com/vllm-project/vllm/pull/41834) — open, stock-deps SM120 path + DSpark
- [arXiv 2606.19348](https://arxiv.org/pdf/2606.19348) — DeepSeek-V4 technical report
