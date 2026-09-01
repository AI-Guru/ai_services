# Qwen3.8-Flash-Next (qwen4exp)

**STATUS: serving on the RTX PRO 6000 at `localhost:11480` via
[`docker-compose.llama-177b-q4-rtx.yml`](docker-compose.llama-177b-q4-rtx.yml)
— UD-Q4_K_XL, llama.cpp master, 95.8 tok/s sustained decode, 3,467 tok/s
prefill at 16K, full 262,144 context, 87.3 of 95.6 GiB VRAM.**

[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is
Qwen's **Qwen4 architecture preview** (`model_type: qwen4_exp`,
`Qwen4ExpForConditionalGeneration`) — not a Qwen3.8 sibling. 176.94B params
total, per llama.cpp:

- **125B MoE** — 512 experts, top-10, `moe_intermediate_size` 640, ~6B active
- **51.2B n-gram / PLE embedding table** — 20,000,000 bigram/trigram entries at layer 2
- **4B MTP** head (absent from the text GGUFs; a separate `MTP/` upload exists)
- 48 layers: Gated DeltaNet on 3 of every 4, **Qwen Sparse Attention** on the 4th
- 262,144 context, multimodal (vision via a separate `mmproj`)

---

## Why a 111 GB checkpoint fits a 96 GiB card

The GGUF is **111.3 GB on disk** and does not fit. It runs because the **26.8
GiB n-gram table is never resident** — llama.cpp's `--lazy-mode` reads its rows
from SSD on demand (PRs
[#27794](https://github.com/ggml-org/llama.cpp/pull/27794) /
[#27837](https://github.com/ggml-org/llama.cpp/pull/27837), merged 2026-08-27
and 08-30). Only ~75 GiB of weights land in VRAM.

Qwen designed for this: *"Embeddings provide a unique axis for parameter scaling
that requires less computation and is more amenable to offloading than MoE...
efficient for memory-constrained accelerators."* Each token gathers only a
handful of rows — roughly 41 KB, ~2 MB/s at 50 tok/s.

**`--lazy-mode` is faster than keeping the table resident, at every depth**
(measured, this box, both on llama.cpp master build 10743):

| prompt tokens | `--lazy-mode on` | `--lazy-mode off` | advantage |
|---:|---:|---:|---:|
| 1,333 | 91.58 | 74.49 | **+23%** |
| 10,667 | 82.07 | 53.87 | **+52%** |
| 85,333 | 56.90 | 51.36 | +11% |
| 133,334 | 46.11 | 41.84 | +10% |

This is the **opposite** of PR #27794's own gemma-4 result (where lazy cost
8-11%), and that PR predicts it: the read delay is significant next to a small
model's token, negligible next to this one's. With only 30 GiB of host RAM the
26.8 GiB table cannot be cached anyway, so forcing residency buys page-cache
thrash and nothing else.

### VRAM model

`VRAM = resident_weights + 33 KiB/token x ctx + ~3.2 GiB fixed`

KV is **33 KiB/token** — 24 KiB main (only 12 of 48 layers are full-attention,
2 KV heads x 256) plus **9 KiB for the QSA indexer**, which is easy to miss.
The full 262,144 context therefore costs only ~8.25 GiB. This predicted
UD-Q3_K_XL within 1.2% and UD-Q4_K_XL within 0.6%.

---

## Measured (2026-09-01, RTX PRO 6000, UD-Q4_K_XL, c=262144)

| metric | build 10656 (`035e22731`) | build 10743 (`8887a48f0`) | change |
|---|---:|---:|---:|
| sustained decode (2048 tok) | 79.5 tok/s | **95.8** | **+21%** |
| short-burst decode (512 tok) | 69-84 | **95.9-100.7** | **+28%** |
| prefill @ 16K | 2,161-2,427 | **3,467** | **+51%** |
| prefill @ 133K | 678 | **2,133** | **+215%** |
| VRAM | 86.0 GiB | 87.3 GiB | +1.3 |

### Throughput vs context depth

Three configs, sequential, idle box, one pass each:

| prompt tok | prefill: old | prefill: master | decode: old | decode: master |
|---:|---:|---:|---:|---:|
| 1,333 | 1,815 | **2,278** | 68.61 | **91.58** |
| 10,667 | 2,615 | **3,651** | 65.35 | **82.07** |
| 42,668 | 1,343 | **3,182** | *(EOS)* | *(EOS)* |
| 85,333 | 902 | **2,609** | 42.73 | **56.90** |
| 133,334 | 678 | **2,133** | 37.57 | **46.11** |

*(EOS: at that prompt the model emits EOS immediately — `predicted_n = 1`,
`stop_type: eos`. Reproducible across all three configs; not a crash.)*

**Prefill at depth improved >3x** — that is
[#28023](https://github.com/ggml-org/llama.cpp/pull/28023), which removed a
transpose from the indexer head summation and explicitly "grows with context".

**Decode improved 23-33% at every depth**, which was not predicted — the known
decode fix is still unmerged, so this came from something else that landed
(likely [#27880](https://github.com/ggml-org/llama.cpp/pull/27880) graph splits,
or [#27941](https://github.com/ggml-org/llama.cpp/pull/27941)).

**The decay with depth is NOT fixed.** Master still falls 95.8 -> 46.1 tok/s
(-52%) from shallow to 133K. The whole curve shifted up; the slope did not
flatten. See "Upstream still open" below.

---

## Quants (Unsloth Dynamic 3.0)

Community quants, not first-party. Unsloth publish KLD and top-1 agreement vs
BF16 — **no task benchmarks for this model**, and they caution that top-1 "is an
argmax on 1 prediction, so it's not really effective on gauging actual
inference." Their preferred **Divergence-300 @32** is not published here.

| quant | GB | GiB | mean KLD | top-1% | VRAM @262K (lazy on) |
|---|---:|---:|---:|---:|---:|
| **UD-Q4_K_XL** ← serving | 111.3 | 103.7 | **0.0447** | **93.5** | **87.3** |
| UD-IQ4_XS | 93.7 | 87.2 | 0.0792 | 91.1 | ~72 |
| UD-Q3_K_XL | 90.0 | 83.8 | 0.0997 | 90.4 | 67.6 |
| UD-IQ3_XXS | 82.0 | 76.3 | 0.1565 | 87.6 | ~61 |
| UD-Q2_K_XL | 78.9 | 73.5 | 0.2133 | 85.2 | ~58 |
| UD-IQ1_M | 74.5 | 69.4 | 0.3022 | 82.4 | ~54 |
| UD-IQ1_S | 72.5 | 67.6 | 0.3751 | 80.2 | ~52 |

**Q4 costs nothing in speed over Q3** (both ~95 tok/s): only 10 of 512 experts
activate per token, so the extra 18 GiB sits in weights that are not read on any
given token — ~0.35 GiB more per token, a fraction of a millisecond.

For scale, Unsloth's Qwen3.5 numbers put *their* Q4_K_XL at mean KLD 0.0137.
**This model quantizes ~3x worse at the same nominal level**, consistent with
512 tiny experts leaving little redundancy. (Cross-model KLD is directional
only — it scales with output entropy.)

The n-gram table is held at **IQ4_NL (~26.8 GiB) in every quant including
IQ1_S** — Unsloth quantize it no lower because of its random access pattern.
So quant choice changes only the resident weights, never the SSD-streamed part.

**"1-bit" is a misnomer**: UD-IQ1_S is **3.28 bpw effective**, with only 32% of
params actually at IQ1_S (57.04B), the rest at IQ4_NL (40.27B) and IQ2_XXS
(23.49B), attention at Q5_K/Q8_0 and GDN at Q6_K.

---

## Traps

1. **`--lazy-mode` requires mmap.** Do not add `--no-mmap` or `-lm none`. The
   qwen3.6 Spark composes use `--no-mmap` — copying that here breaks the one
   thing that makes this model fit.
2. **Two competing llama.cpp implementations existed.**
   [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) (Unsloth's,
   merged 08-27) uses `per_layer_token_embd`;
   [#27739](https://github.com/ggml-org/llama.cpp/pull/27739) (never merged)
   used `blk.N.ple_ngram_embd` with different metadata keys. The Unsloth GGUFs
   require #27742's naming, and the tensor is loaded `TENSOR_NOT_REQUIRED` —
   **a mismatch silently skips 51B params and still emits fluent text.**
3. **The host `nvcc` is 12.0 and cannot target SM_120** (needs >=12.8), so
   Unsloth's copy-paste host build fails here. Build in the CUDA 13 container —
   see [`Dockerfile.llama-qwen4exp`](Dockerfile.llama-qwen4exp).
4. **Old builds abort at deep context.** `rms_norm_f32` exceeds the CUDA
   `gridDim.y` limit (65535) at `n_kv` 262144
   ([#27901](https://github.com/ggml-org/llama.cpp/issues/27901)) — not OOM.
   Fixed by #27941 (merged 09-01). A pre-09-01 build at `-c 262144` is a
   latent crash.
5. **`reasoning_effort` accepts only `xhigh` (default), `medium`, `low`.**
   `"none"` and `"default"` return **HTTP 500** with a Jinja exception, despite
   Unsloth's docs listing `none`. Pass via
   `chat_template_kwargs: {"reasoning_effort": "medium"}`.
6. **Thinking tokens spend `max_tokens`.** At `max_tokens: 40` the API returns
   `content: ""` with `completion_tokens: 40` — not an error, just an empty
   string, because reasoning consumed the whole budget. Effort levels are
   adaptive, not fixed budgets: on an easy prompt `xhigh` was the *tersest*
   of the three. `--reasoning-budget N` caps thinking without capping the answer.
7. **"Truncated response" is almost always the client's `max_tokens`.**
   Pi-Agent capped at exactly 16384 twice; the server had never truncated
   anything (`truncated = 0` on every request) and generated 17,000 on demand
   when asked. Check the client before the context. A one-file game like the
   platformer prompt genuinely needs 30-60K tokens.
8. **Long sessions degrade** — see the depth table. Restarting resets it.

---

## Files

| File | What |
|---|---|
| [`docker-compose.llama-177b-q4-rtx.yml`](docker-compose.llama-177b-q4-rtx.yml) | Production config. Port 11480, served as `qwen3.8-flash-next` |
| [`Dockerfile.llama-qwen4exp`](Dockerfile.llama-qwen4exp) | CUDA 13 / SM_120 build. `LLAMA_REF=master` (default) or `pr/<N>` |

Weights are **not** in this repo — 111.3 GB at
`/home/despara/models/qwen3.8-flash-next/UD-Q4_K_XL/`:

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --local-dir /home/despara/models/qwen3.8-flash-next --include "*UD-Q4_K_XL*"
```

---

## Upstream still open

| PR | State | Claim |
|---|---|---|
| [#27992](https://github.com/ggml-org/llama.cpp/pull/27992) | closed **draft** | O(log n) n-gram predecessor lookup — **2.72x decode at 240K** on 2xL40S |
| [#28136](https://github.com/ggml-org/llama.cpp/pull/28136) | open | direct reads for the lazy PLE table, >2x prefill on GB10 |
| [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) / [#28097](https://github.com/ggml-org/llama.cpp/pull/28097) | open | NextN/MTP draft head (`--spec-type draft-mtp`) |

**#27992 is the one that matters here** — it targets exactly the decay measured
above. `get_prev_tokens()` scans every used cell to resolve n-gram predecessors;
at 88K that is ~20 million iterations per token, and the author saw the GPU
drawing 315W instead of 570W. It is closed **as a draft** proof-of-concept, not
queued to land, so cherry-picking is the only near-term route.

MTP is the largest single win still unavailable: a dropped port
([#28104](https://github.com/ggml-org/llama.cpp/pull/28104)) measured **+50%
decode at 70K** with 70-80% draft acceptance, growing with context. Unsloth now
publish the draft head (`MTP/`) and a vision projector (`mmproj-*.gguf`) — both
absent when this was first brought up.

## Not configured (deliberately)

`--cache-type-k/v q8_0` (untested against the QSA + GDN hybrid cache), `-fa on`
(never benchmarked either way here), `--spec-type draft-mtp` (llama.cpp side not
merged). `--jinja` is already the default and is what parses
`chat_template_kwargs`.
