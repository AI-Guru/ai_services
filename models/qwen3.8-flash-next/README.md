# Qwen3.8-Flash-Next (qwen4exp)

**STATUS: serving on the RTX PRO 6000 at `localhost:11480` via
[`docker-compose.llama-177b-q4-mtp-rtx.yml`](docker-compose.llama-177b-q4-mtp-rtx.yml)
— UD-Q4_K_XL, **vision and MTP both on**, full 262,144 context, 91.7 of 95.6 GiB
VRAM on build 11112 (master + #28243, since 2026-09-22). Measured on real traffic: **≈1.35–1.4x from MTP**, throughput 42–146 tok/s
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
| + vision | 88.4 GiB | 7.2 |
| + MTP | 91.7 GiB | 3.9 |
| + vision + MTP | 92.8 GiB | 2.8 |
| **+ vision + MTP, builds 10945 / 11112** ← serving | **91.7 GiB** | **3.9** |

The last row is `nvidia-smi` at idle (93,883 MiB, against 95,419 MiB for the same
config on build 10802) — a different method from the rows above, so compare it
with 10802's 95,419, not with 92.8. The 1.5 GiB came from #28330; see the A/B.

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

temp 1.0 / top-p 0.95 / top-k 20, `n_predict 256`, `ignore_eos`, mean of 2 runs.
**This table uses `ignore_eos`, which Trap 10 shows inflates MTP**; read its MTP
column as an upper bound. The A/B below uses the corrected method.

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
the curve; it does not flatten it. This was blamed on the n-gram predecessor scan
(#27992), but its replacement [#28040](https://github.com/ggml-org/llama.cpp/pull/28040)
merged on 2026-09-01 and is in every MTP build here — the decay persists (below),
so it is something else.

### Build 10945 vs 10802 — A/B on real text (2026-09-13)

10945 is GitHub's merge of #28243 into master `ae9afff8` (pinned `c83604f0b`),
i.e. the MTP fork rebased; 10802 is the fork's own `d1a92352c`. Identical flags
to production, one image at a time on a private port. Prompt: this repo's
markdown sliced to depth plus a "summarize it" question; temp 1.0 / top-p 0.95 /
top-k 20, 512 tokens, **no `ignore_eos`, `n_gen` asserted** — all 32 runs reached
512 — and `cache_prompt: false`. Order old → new → old, compared against the
*second* old run so both saw an equally warm page cache. (The bias turned out
negligible: cold vs warm prefill agreed within ~1 %.)

**Metric: ms per target forward pass** = `predicted_ms / (n_gen − draft_accepted)`.
tok/s under MTP at temp 1.0 tracks acceptance, not the engine: at 250K the same
build read 41.9 and 56.3 tok/s at 41 % vs 68 % acceptance — and 53.15 vs 53.82
ms/pass. Runs at the same depth agree within ~1 ms on this metric.

| ctx | ms/pass 10802 | ms/pass **10945** | ratio | prefill 10802 | prefill 10945 |
|---:|---:|---:|---:|---:|---:|
| 2K | 20.87 | 20.85 | 0.999 | 682 | 688 |
| 4K | 20.26 | 20.43 | 1.008 | 1,228 | 1,209 |
| 8K | 20.63 | 21.01 | 1.019 | 1,726 | 1,744 |
| 16K | 21.08 | 21.78 | 1.033 | 1,713 | 1,717 |
| 32K | 23.74 | 23.66 | 0.996 | 1,594 | 1,590 |
| 64K | 27.85 | 28.09 | 1.009 | 1,568 | 1,576 |
| 128K | 37.51 | 37.05 | 0.988 | 1,444 | 1,444 |
| 250K | 53.22 | 53.48 | 1.005 | 1,092 | 1,091 |

**Speed-neutral.** 148 upstream commits changed nothing on this config's hot
path. The rebase is a **VRAM win: 93,883 vs 95,419 MiB idle, 1.5 GiB freed**,
headroom 2.4 → 3.9 GiB. That is
[#28330](https://github.com/ggml-org/llama.cpp/pull/28330) (the QSA indexer no
longer allocates a V cache it never reads). Its PR reported ~6 GiB at 262K; here
the whole indexer is 9 KiB/token, so its V half at 262K is ~1.1 GiB — the figure
does not transfer to this model.

**The depth decay is per-pass engine cost, not speculation.** Tokens per forward
pass stays 2.6–3.0 at every depth, so MTP helps just as much at 250K; ms/pass is
flat to 16K and then rises 2.6x by 250K.

### Build 11112 vs 10945 — A/B on real text (2026-09-22)

11112 is GitHub's merge of #28243 head `6fcaa16f` into master `ec5a12b8`
(pinned `127368d8`); 10945 is the previous production build. Same method as
above, order 10945 → 11112 → 10945, `benchmarks/depth_ab.py` + `analyze_ab.py`.

| ctx | ms/pass 10945 (run 1 / run 2) | ms/pass **11112** | ratio vs faster baseline | prefill 10945 (1 / 2) | prefill **11112** |
|---:|---:|---:|---:|---:|---:|
| 2K | 21.12 / 21.17 | 21.28 | 1.008 | 675 / 675 | 664 |
| 4K | 20.74 / 20.90 | 20.27 | 0.977 | 1,203 / 1,206 | 797† |
| 8K | 20.93 / 21.03 | 20.38 | 0.974 | 1,748 / 1,718 | 1,772 |
| 16K | 21.91 / 21.85 | 21.14 | 0.968 | 1,725 / 1,705 | 1,736 |
| 32K | 24.56 / 24.59 | 23.48 | 0.956 | 1,611 / 1,585 | 1,633 |
| 64K | 28.32 / 28.25 | 27.33 | 0.965 | 1,587 / 1,566 | 1,698 |
| 128K | 37.61 / 39.12 | 36.00 | 0.957 | 1,445 / 1,080 | **1,748** |
| 250K | 54.23 / 56.56 | 52.08 | 0.960 | 1,090 / 783 | **1,526** |

† 4K prefill on 11112 read 763/831 on both runs against ~1,200 everywhere else
— an outlier at one depth, not reproduced at 2K or 8K, cause unknown.

**Decode: −2.5 to −4.5 % per forward pass from 4K up**, against whichever
baseline run was faster. That is the fused hyper-connection and norm ops
([#28901](https://github.com/ggml-org/llama.cpp/pull/28901),
[#28896](https://github.com/ggml-org/llama.cpp/pull/28896)), not sparse FA —
the 4-token MTP verify batch cannot reach the sparse kernel, and tokens per pass
is unchanged. Small, but the first build-to-build decode gain since MTP landed,
and it is not acceptance noise.

**Prefill at depth is the real win: +21 % at 128K and +40 % at 250K** vs the
faster baseline run (1.6–1.9x vs the slower one). Prefill batches are wide
enough for [#28770](https://github.com/ggml-org/llama.cpp/pull/28770)'s
8-column sparse-FA tiles. A cold 250K prompt now prefills in ~164 s instead of
~229 s. Nothing below 64K, as the PR's own curve predicts.

**The unexplained variance showed up again, mid-sweep, in the second baseline
run only:** from its 128K/run 2 onward, prefill dropped to 770–830 and ms/pass
rose ~4 % (39.12 vs 37.61, 56.56 vs 54.23) with the flags, prompts and page cache
unchanged and the GPU at 57 °C / 2,617 MHz — the same phenomenon as Trap 11, now
caught inside a controlled run. Both baseline runs are shown for that reason,
and the ratios above use the faster one so the gain is not overstated.

VRAM unchanged: 93,861 vs 93,883 MiB.

**`--spec-draft-n-max 7` re-tested on 11112, because an 8-token verify batch is
the one MTP shape the sparse-FA kernel specializes.** It loses: ms/pass 1.25–1.43x
worse at 64K–250K, acceptance 25–39 %, tok/s 0.84x / 0.90x / 1.06x at 64K / 128K
/ 250K, and it costs **+1.8 GiB VRAM** (95,663 MiB). Sparse FA does not rescue a
wide verify batch. n-max 3 stays.

---

## MTP (speculative decoding)

[`docker-compose.llama-177b-q4-mtp-rtx.yml`](docker-compose.llama-177b-q4-mtp-rtx.yml)
— **this is what production runs.**

**Needs #28243, not plain mainline.** Master has no MTP graph for qwen4exp —
#28243 is an open draft, #27836/#28097 open, #28104/#27842/#27956 dropped. Mainline *accepts*
`--spec-type draft-mtp` (it exists for other archs) and then **silently ignores
the head**: baseline speed, no error. Always confirm with the log line
`draft acceptance = 0.73939 (610 accepted / 825 generated), mean len = 2.48`.
No acceptance line = no speculation.

**Since 2026-09-22 production runs build 11112:** GitHub's merge of #28243 head
`6fcaa16f` into master `ec5a12b8`, pinned as `127368d8` and tagged
`llama.cpp-qwen4exp:mtp-127368d8`. −3 % decode per pass, +21–40 % prefill at
128K–250K over 10945 (A/B above); vision and 4/4 tool scenarios re-verified.
Rollback: `:mtp-c83604f0` (10945, 2026-09-13, speed-neutral vs the fork, 1.5 GiB
less VRAM) or `:mtp` (the fork's own `d1a92352c`, build 10802).

`d1a92352c` (`danielhanchen/llama.cpp` branch `qwen4exp/mtp`) is what Unsloth's
MTP guide points at. Worth +4 to +8 % over
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

**Update 2026-09-13: that hypothesis no longer stands as written.** #27992 was
superseded by [#28040](https://github.com/ggml-org/llama.cpp/pull/28040), merged
2026-09-01 — and `d1a92352c`, the build that served all 178 requests, already
contains it (52 commits ahead, 0 behind). The O(log n) lookup was live the whole
time the 3x spread was measured, so the whole-cache scan cannot be its cause.
`kv_unified` interference through some *other* path is still untested.

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
| [`docker-compose.llama-177b-q4-mtp-rtx.yml`](docker-compose.llama-177b-q4-mtp-rtx.yml) | **Production.** Vision + MTP. Carries `restart: always` |
| [`docker-compose.llama-177b-q4-rtx.yml`](docker-compose.llama-177b-q4-rtx.yml) | Fallback / multi-user. Vision on, MTP off. 88.4 GiB. Same port — mutually exclusive |
| [`Dockerfile.llama-qwen4exp`](Dockerfile.llama-qwen4exp) | CUDA 13 / SM_120. `LLAMA_REPO` + `LLAMA_REF` select the commit; production pins the #28243 merge SHA |
| [`benchmarks/depth_ab.py`](benchmarks/depth_ab.py) / [`analyze_ab.py`](benchmarks/analyze_ab.py) | The depth-sweep A/B used above: real prompts, no `ignore_eos`, `n_gen` asserted, ms per forward pass |

Weights are **not** in this repo (~115 GB at
`/home/despara/models/qwen3.8-flash-next/`):

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --local-dir /home/despara/models/qwen3.8-flash-next \
  --include "*UD-Q4_K_XL*" "mmproj-F16.gguf" "MTP/*shared-Q8_0*"
```

---

## Upstream status (checked 2026-09-22)

Merged into master since build 10945 (`ae9afff8`, 2026-09-12):

| PR | Merged | Claim | For this config |
|---|---|---|---|
| [#28770](https://github.com/ggml-org/llama.cpp/pull/28770) CUDA sparse FA for qwen4 | 2026-09-20 | tg 1.03–1.18x, pp 1.08–1.26x at 10K–100K (DGX Spark); auto-on once KV ≥ max(4096, 2·gather) | Only 1- and 8-token query batches are specialized (`ncols1 == 1 \|\| 8`); MTP n-max 3 verifies 4 tokens per pass → dense kernel. Author deferred 2/4 to follow-up work. **Measured below** |
| [#28901](https://github.com/ggml-org/llama.cpp/pull/28901) hyper-connection fused ops | 2026-09-16 | pp 1.14x, tg 1.02x (Spark); CUDA included | Applies |
| [#28896](https://github.com/ggml-org/llama.cpp/pull/28896) rms_norm + mul fusion | 2026-09-14 | pp +3 % | Applies |
| [#28040](https://github.com/ggml-org/llama.cpp/pull/28040) O(log n) `get_prev_tokens` | 2026-09-01 | +3.9 % at 55K, +12 % at 132K, this card and quant | In every MTP build since `d1a92352c` |
| [#28330](https://github.com/ggml-org/llama.cpp/pull/28330) indexer skips V cache | 2026-09-10 | ~6 GiB at 262K, reported | Measured 1.5 GiB here (build 10945) |

Still open:

| PR | State | Claim | For this config |
|---|---|---|---|
| [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) qwen4exp MTP | open, **out of draft**, mergeable clean, head `6fcaa16f` (2026-09-21) | the MTP graph every build here is made from | 3 commits since `d1a92352c`: conflict fixes + review comments |
| [#29166](https://github.com/ggml-org/llama.cpp/pull/29166) QSA per-block bias with a unified cache | draft, depends on #28699 | **correctness**: with `kv_unified` and >1 live sequence, block visibility is indexed by block number, not id — "the model stops seeing its last messages and reports the input as empty"; 2/4 → 4/4 | Production runs 4 slots on a unified cache and all 4 see traffic. If a request ever came back claiming an empty input under concurrent use, this is the cause. Not cherry-pickable until #28699 lands |
| [#29030](https://github.com/ggml-org/llama.cpp/pull/29030) direct reads for lazy rows (supersedes [#28136](https://github.com/ggml-org/llama.cpp/pull/28136)) | open; ggerganov wants the problem stated better, design moving to libllama | cold pp2048 277 → 484 tok/s (+75 %), +11 % at 8K, tg unchanged; output byte-identical | Would cut cold TTFT on short prompts; nothing at depth |
| [#28213](https://github.com/ggml-org/llama.cpp/pull/28213) gather-based QSA decode | open, untouched since 2026-09-10 | +6 / +19 / +50 % at 31K / 62K / 130K (2x A6000, no MTP) | Gates on `n_tokens == n_stream` → no gain under MTP; multi-token follow-up [#28349](https://github.com/ggml-org/llama.cpp/pull/28349) closed unmerged |
| [#28699](https://github.com/ggml-org/llama.cpp/pull/28699) pooled-key cache for the QSA indexer | draft | +9.3 % decode at 63K / 114K | Vision M-RoPE assertion still unresolved, plus a new cross-sequence bug under a unified cache |
| [#28751](https://github.com/ggml-org/llama.cpp/pull/28751) no scheduler re-reserve on `causal_attn` toggle | open | 1.27–7.6x multi-image prefill | Vision only |

**Every open decode-side win still conflicts with MTP, vision, or the unified
cache.** Rebasing onto master remains the only lever that applies as-is.
