# Qwen3.8-Flash-Next (qwen4exp)

**STATUS: serving on the RTX PRO 6000 at `localhost:11480` via
[`docker-compose.llama-177b-q4-mtp-rtx.yml`](docker-compose.llama-177b-q4-mtp-rtx.yml)
— UD-Q4_K_XL, **vision and MTP both on**, full 262,144 context, 93.2 of 95.6 GiB
VRAM. Measured on real traffic: **≈1.35–1.4x from MTP**, throughput 42–146 tok/s
depending on workload. The non-MTP compose remains as the fallback and for
multi-user serving.**

[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is
Qwen's **Qwen4 architecture preview** (`model_type: qwen4_exp`) — not a Qwen3.8
sibling. 176.94B params total, per llama.cpp:

- **125B MoE** — 512 experts, top-10, `moe_intermediate_size` 640, ~6B active
- **51.2B n-gram / PLE embedding table** — 20,000,000 bigram/trigram entries at layer 2
- **4B MTP** head (separate `MTP/` upload; see below)
- 48 layers: Gated DeltaNet on 3 of every 4, **Qwen Sparse Attention** on the 4th
- 262,144 context, multimodal (vision + video via a separate `mmproj`)

---

## Why a 111 GB checkpoint fits a 96 GiB card

llama.cpp places the 26.8 GiB n-gram table **host-side, not in VRAM**, and has
done so since #27742 with no flag involved. Only ~75 GiB of weights reach the
card. (Verified: a build predating `--lazy-mode` logged
`CPU_Mapped model buffer size = 28110 MiB` and ran at 86.0 GiB.)

**`--lazy-mode` is not what makes it fit.** VRAM is the same either way. What it
changes is *how* the host-side table is read: it skips the prefetch and marks the
mapping `MADV_RANDOM`, so rows stream from SSD instead of the loader trying to
cache 26.8 GiB in ~25 GiB of page cache (PRs
[#27794](https://github.com/ggml-org/llama.cpp/pull/27794) /
[#27837](https://github.com/ggml-org/llama.cpp/pull/27837)). A decode-speed and
stability win, not a capacity one.

**Lazy beats resident at every depth** (both on master build 10743):

| prompt tokens | `--lazy-mode on` | `off` | advantage |
|---:|---:|---:|---:|
| 1,333 | 91.58 | 74.49 | **+23%** |
| 10,667 | 82.07 | 53.87 | **+52%** |
| 133,334 | 46.11 | 41.84 | +10% |

The opposite of PR #27794's gemma-4 result, and that PR predicts it: the read
delay is significant next to a small model's token, negligible next to this one's.
With 30 GiB of host RAM the table cannot be cached anyway.

### VRAM model

`VRAM = resident_weights + 33 KiB/token x ctx + ~3.2 GiB fixed`

KV is **33 KiB/token** — 24 KiB main (only 12 of 48 layers are full-attention,
2 KV heads x 256) plus **9 KiB for the QSA indexer**, which is easy to miss.
The full 262,144 context costs ~8.25 GiB. Predicted UD-Q3_K_XL within 1.2% and
UD-Q4_K_XL within 0.6%.

### Measured footprints (262,144 context)

| configuration | VRAM | free |
|---|---:|---:|
| model only | 87.3 GiB | 8.3 |
| **+ vision** ← serving | **88.4 GiB** | **7.2** |
| + MTP | 91.7 GiB | 3.9 |
| + vision + MTP | 92.8 GiB | 2.8 |

---

## Throughput

**Sustained (build 10743, `--lazy-mode on`):** 95.8 tok/s decode, 3,467 tok/s
prefill at 16K. Against a build pinned to PR #27742's head a week earlier:

| metric | build 10656 | build 10743 | change |
|---|---:|---:|---:|
| sustained decode | 79.5 | **95.8** | +21% |
| prefill @ 16K | 2,161-2,427 | **3,467** | +51% |
| prefill @ 133K | 678 | **2,133** | +215% |

Splitting the A/B three ways separates the causes cleanly by metric: **prefill is
entirely upstream ([#28023](https://github.com/ggml-org/llama.cpp/pull/28023));
`--lazy-mode` contributes +0.1% to it.** Decode is ~10-20% upstream plus another
~10-23% from streaming the table.

### Depth sweep — with and without MTP

temp 1.0 / top-p 0.95 / top-k 20, `n_predict 256`, `ignore_eos`, mean of 2 runs:

| ctx | decode off | decode **MTP** | factor | prefill off | prefill **MTP** | factor |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 97.38 | **138.35** | 1.42x | 2,982 | 2,610 | 0.88x |
| 4K | 95.22 | **128.06** | 1.34x | 3,590 | 3,231 | 0.90x |
| 8K | 85.64 | **120.47** | 1.41x | 3,556 | 3,259 | 0.92x |
| 16K | 89.62 | **109.09** | 1.22x | 3,376 | 3,107 | 0.92x |
| 32K | 78.41 | **101.94** | 1.30x | 3,066 | 2,827 | 0.92x |
| 64K | 65.88 | **71.83** | 1.09x | 2,534 | 2,341 | 0.92x |
| 128K | 49.64 | **56.72** | 1.14x | 1,811 | 1,678 | 0.93x |
| 256K | 32.72 | **42.21** | 1.29x | 1,141 | 1,060 | 0.93x |

**Throughput falls ~two thirds from 2K to 256K, with MTP and without.** MTP lifts
the curve; it does not flatten it. That is
[#27992](https://github.com/ggml-org/llama.cpp/pull/27992), still unmerged.

---

## MTP (speculative decoding)

[`docker-compose.llama-177b-q4-mtp-rtx.yml`](docker-compose.llama-177b-q4-mtp-rtx.yml)
— **this is what production runs.**

**Needs a fork, not mainline.** Upstream has no MTP graph for qwen4exp —
#27836/#28097 open, #28104/#27842/#27956 dropped. Mainline *accepts*
`--spec-type draft-mtp` (it exists for other archs) and then **silently ignores
the head**: baseline speed, no error. Always confirm with the log line
`draft acceptance = 0.73939 (610 accepted / 825 generated), mean len = 2.48`.
No acceptance line = no speculation.

Built from `danielhanchen/llama.cpp` branch `qwen4exp/mtp` (verified at
`d1a92352c`), which is what Unsloth's MTP guide points at. Worth +4 to +8 % over
the older `unslothai` PR #144 build (`586b15ef8`), entirely inside the MTP path —
the non-MTP baseline is unchanged at 96.07 vs 95.53 tok/s greedy.

### It is lossless — verified in the source, not assumed

`common_sampler_sample_and_accept_n` in `common/sampling.cpp`:

```c
for (; i < draft.size(); i++) {
    const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
    result.push_back(id);        // ALWAYS the target model's own sample
    if (draft[i] != id) break;   // the draft only decides whether to continue
}
```

The emitted token is always the target's; the draft is never copied into the
output. At temp 0 the result is bit-identical, and at temp 1.0 every token comes
from the same distribution as unspeculated decoding. Acceptance changes only how
many forward passes are saved. **Quality risk here is the quant, not MTP.**

### `--spec-draft-n-max 3`, measured — not the 5 the vendor guide recommends

| n-max | greedy | temp 1.0 | acceptance | factor vs no-MTP |
|---|---:|---:|---:|---:|
| 2 | 139.18 | 135.73 | 67.7 % | 1.38x |
| **3** | **142.87** | **136.67** | 57.8 % | **1.39x** |
| 5 | 116.64 | 117.50 | 43.9 % | 1.20x |
| 8 | 96.31 | 93.33 | 31.6 % | **0.95x** |

5 is slower than the 2 we started with; 8 is a net loss against no speculation at
all. The older MTP README in the model repo ("2 is a good default") was closer
than the newer guide. Note `mean len` tops out at 4.00 with n-max 3 — three
drafted tokens plus the target's own.

### Measured on real traffic — 178 requests, >70,000 drafts

| | |
|---|---|
| **pooled factor** | **≈1.35–1.4x**, stable across everything |
| pooled acceptance | ~58 % |
| **throughput range** | **42 – 146 tok/s** |
| acceptance range | 37.8 – 100 % |
| context spanned | 632 – 230,000 |
| errors / truncation / OOM | **none**, at 2.4 GiB headroom |

**The factor is the number to plan with. The absolute figure is not.** Throughput
varies by more than 3x under an unchanged configuration, and prefill varies in the
same proportion (185 – 2,018 tok/s), which rules out content and acceptance as the
cause. Eight explanations were tested and rejected: answer length, session
duration, a "phase", context depth, KV-pool occupancy, concurrency, clock
throttling, and page cache. The one surviving hypothesis — that under
`kv_unified` a slot holding a large foreign context slows every other request,
via the whole-cache scan in `get_prev_tokens()` that
[#27992](https://github.com/ggml-org/llama.cpp/pull/27992) fixes — could not be
checked without `-v` or `--slots`. If it holds, `--parallel 1` would be
substantially faster for single-user work.

**Do not benchmark speculation with `ignore_eos`.** It was introduced here to stop
the EOS artifact (see Traps) from corrupting throughput cells, and it does that —
but forcing generation past the natural end produces degenerate repetition that a
draft head predicts perfectly. Acceptance read **94.5 % on filler text and 100 %
on a truncated book, where the model emitted nothing but `0` characters**, which
inflated MTP at 130K to ~110 tok/s and 2.3x. Both figures were artifacts. Real
prompts without `ignore_eos` give 55–70 %.

**Single stream only.** Unsloth measure MTP as a net loss (~0.81-0.87x) at
concurrency 8. Not re-measured here.

---

## Vision

On by default in the production compose — the projector costs only **~1.1 GiB**.

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --local-dir /home/despara/models/qwen3.8-flash-next --include "mmproj-F16.gguf"
```

Server reports `modalities {vision: true, video: true, audio: false}`; **video is
untested.** A 720x720 JPEG costs ~580 prompt tokens and answers in ~5 s, with **no
measurable cost to text throughput** (95.33 tok/s with the projector loaded).

Verified against the actual images rather than taken on trust: it described a
photo and a synthetic render correctly. The second test is the meaningful one —
a famous film still could be recalled from text training; a generated image cannot.

Vision and MTP coexist (an image request answered correctly while the log still
showed `draft acceptance = 0.58743`), but together they leave only 2.8 GiB, and
per-image buffers under *concurrent* vision requests were never measured.

---

## Tool calling

`models/shared/test_tools.py` against this endpoint: **12/12 across three passes**,
all four scenarios each time.

| pass | single | parallel | chained | multi-parallel |
|---|---|---|---|---|
| 1 | 944 ms | 895 ms | 3 turns | 1,815 ms |
| 2 | 998 ms | 1,003 ms | 3 turns | 2,169 ms |
| 3 | 1,005 ms | 1,331 ms | 3 turns | 1,552 ms |

Parallel dispatch is real (3 mixed calls in one response), chained state survives
(weather 22°C -> `22 * 3.14` -> 69.08 -> `finish_reason=stop`), and argument
schemas were clean throughout.

This matters because **Unsloth publish no task benchmarks for this model** and say
themselves that top-1 "is an argmax on 1 prediction, so it's not really effective
on gauging actual inference." Their preferred Divergence-300 @32 is not published
here either. Tool calling is the only capability number measured on this hardware.

---

## Quants (Unsloth Dynamic 3.0)

Community quants, not first-party. KLD and top-1 agreement vs BF16 only.

| quant | GB | GiB | mean KLD | top-1% |
|---|---:|---:|---:|---:|
| **UD-Q4_K_XL** ← serving | 111.3 | 103.7 | **0.0447** | **93.5** |
| UD-IQ4_XS | 93.7 | 87.2 | 0.0792 | 91.1 |
| UD-Q3_K_XL | 90.0 | 83.8 | 0.0997 | 90.4 |
| UD-Q2_K_XL | 78.9 | 73.5 | 0.2133 | 85.2 |
| UD-IQ1_S | 72.5 | 67.6 | 0.3751 | 80.2 |

**Q4 costs nothing in speed over Q3** (both ~95 tok/s): only 10 of 512 experts
activate per token, so the extra 18 GiB sits in weights not read on any given
token — ~0.35 GiB more per token.

Unsloth's Qwen3.5 numbers put *their* Q4_K_XL at mean KLD 0.0137, so **this model
quantizes ~3x worse at the same nominal level** — consistent with 512 tiny experts
leaving little redundancy. (Cross-model KLD is directional only.)

The n-gram table is **IQ4_NL (~26.8 GiB) in every quant including IQ1_S** — never
quantized lower, because of its random access pattern. Quant choice changes only
the resident weights.

**"1-bit" is a misnomer**: UD-IQ1_S is **3.28 bpw effective**, with only 32% of
params actually at IQ1_S.

---

## Traps

1. **`--lazy-mode` requires mmap.** Never add `--no-mmap` or `-lm none`. The
   qwen3.6 Spark composes use `--no-mmap` — copying that here breaks it.
2. **Two competing llama.cpp implementations existed.** #27742 (merged) uses
   `per_layer_token_embd`; #27739 (never merged) used `blk.N.ple_ngram_embd`.
   The tensor loads `TENSOR_NOT_REQUIRED` — **a mismatch silently skips 51B
   params and still emits fluent text.**
3. **Mainline silently ignores the MTP head.** See the MTP section.
4. **The host `nvcc` is 12.0 and cannot target SM_120** (needs >=12.8). Build in
   the CUDA 13 container.
5. **Old builds abort at deep context.** `rms_norm_f32` exceeds the CUDA
   `gridDim.y` limit at `n_kv` 262144
   ([#27901](https://github.com/ggml-org/llama.cpp/issues/27901)) — not OOM.
   Fixed by #27941 (merged 2026-09-01).
6. **`reasoning_effort` accepts only `xhigh` (default), `medium`, `low`.**
   `"none"` returns **HTTP 500** despite Unsloth's docs listing it.
7. **Thinking tokens spend `max_tokens`.** At `max_tokens: 40` the API returns
   `content: ""` with `completion_tokens: 40`. Effort levels are adaptive, not
   fixed budgets: on an easy prompt `xhigh` was the *tersest* of the three.
   `--reasoning-budget N` caps thinking without capping the answer.
8. **"Truncated response" is almost always the client's `max_tokens`.** Pi-Agent
   capped at exactly 16384 twice; the server had never truncated anything
   (`truncated = 0` on every request) and produced 17,000 on demand when asked.
   A one-file game like the platformer prompt needs 30-60K tokens.
9. **Benchmark artifact, part one:** at temp 1.0 on filler prompts the model
   sometimes emits EOS immediately (`predicted_n = 1`), which silently corrupts an
   averaged throughput cell — `(75.08 + 0.00)/2` once read as 37.54 tok/s. Use
   `ignore_eos` for throughput runs **and assert `n_gen`**.
10. **Benchmark artifact, part two:** but `ignore_eos` then destroys any
   *speculation* measurement — see the MTP section. Forced generation past the
   natural end degenerates into repetition a draft head predicts perfectly.
   Benchmark MTP on real prompts, without `ignore_eos`.
11. **A single throughput number for this model is misleading.** Real traffic
   spans 42–146 tok/s at identical configuration, prefill 185–2,018 tok/s, with
   the cause unresolved after eight tested hypotheses. Quote the MTP *factor*
   (stable) or a measured range, never a point value.

---

## Files

| File | What |
|---|---|
| [`docker-compose.llama-177b-q4-rtx.yml`](docker-compose.llama-177b-q4-rtx.yml) | **Production.** Vision on, MTP off. 88.4 GiB |
| [`docker-compose.llama-177b-q4-mtp-rtx.yml`](docker-compose.llama-177b-q4-mtp-rtx.yml) | + MTP. 92.8 GiB, only 2.8 free. Same port — mutually exclusive |
| [`Dockerfile.llama-qwen4exp`](Dockerfile.llama-qwen4exp) | CUDA 13 / SM_120. `LLAMA_REPO` + `LLAMA_REF` select upstream master or the Unsloth MTP fork |

Weights are **not** in this repo (~115 GB at
`/home/despara/models/qwen3.8-flash-next/`):

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --local-dir /home/despara/models/qwen3.8-flash-next \
  --include "*UD-Q4_K_XL*" "mmproj-F16.gguf" "MTP/*shared-Q8_0*"
```

---

## Upstream still open

| PR | State | Claim |
|---|---|---|
| [#27992](https://github.com/ggml-org/llama.cpp/pull/27992) | closed **draft** | O(log n) n-gram lookup — **2.72x decode at 240K** on 2xL40S |
| [#28136](https://github.com/ggml-org/llama.cpp/pull/28136) | open | direct reads for the lazy PLE table, >2x prefill on GB10 |
| [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) / [#28097](https://github.com/ggml-org/llama.cpp/pull/28097) | open | NextN/MTP upstream (the fork has it already) |

**#27992 targets exactly the decay in the depth table.** `get_prev_tokens()` scans
every used cell to resolve n-gram predecessors; at 88K that is ~20 million
iterations per token, with the GPU drawing 315W instead of 570W. It is closed **as
a draft** proof-of-concept, so cherry-picking is the only near-term route.
