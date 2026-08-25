# Qwen3.8 Family

**STATUS: released 2026-08-14. W4A4 NVFP4 + MTP-2 + `probabilistic` is the
production config — 1,159 tok/s aggregate at concurrency 30, 108.7 tok/s single
stream (and 122.1 greedy, the best in the table), tools 4/4, vision, 43K-token recall, temp-0 determinism, and zero errors
across 5 concurrency scenarios and mixed text+image load to concurrency 32.**
`Qwen/Qwen3.8-27B` and `Qwen/Qwen3.8-27B-FP8` are both first-party and public
(Apache 2.0), but neither is the fast path on this card — see the table below.

**Currently serving on the RTX PRO 6000 at `localhost:11484`:**
`docker-compose.vllm-27b-nvfp4-gittensor-mtp-prob-rtx.yml`.

⚠️ **That is NOT the fastest single-stream config, and the difference matters.**
There are two answers here and you have to pick on concurrency:

| | single stream | at concurrency 30 | KV pool | image decode @32 |
|---|---|---|---|---|
| DSpark-7 + `probabilistic` | **111.8 tok/s** | 891 tok/s | 967,870 | 18.4 tok/s |
| **MTP-2 + `probabilistic`** ← production | 108.7 tok/s | **1,159 tok/s** | **1,557,969** | **36.1 tok/s** |

The single-stream table below is sorted by single-stream speed, as it must be.
**Read [Production selection](#production-selection-concurrency-changes-the-answer-2026-08-18)
before deploying any of it** — the config that wins there ranks 5th here.

📊 **[`20260818-qwen38-27b-speculation-crossover.html`](20260818-qwen38-27b-speculation-crossover.html)**
— the whole investigation as one page: ten interactive charts covering the
concurrency inversion, the memory-vs-compute arithmetic behind it, the
draft-length curve, the mixed text+image gap, KV-pool cost, per-position
acceptance decay, and the eight things that did not work. Self-contained; open
it in a browser.

The gain came from three independent changes, none of which is "use NVFP4 +
MTP" — that was already being done:

| Change | Effect | Why it was not obvious |
|---|---|---|
| **W4A4** checkpoint instead of W4A16 | 58.7 -> 77.1 tok/s (**+31%**) | Both repos are called "NVFP4". Only one puts *activations* in 4-bit, and only that one lights up the FP4 tensor cores |
| **DSpark** block drafter instead of the built-in MTP head | 77.1 -> 102 tok/s | MTP got *worse* when the target got faster — a harder-quantized target diverges further from its own BF16 draft head |
| **`draft_sample_method: probabilistic`** | 102 -> **111.8 tok/s** (**+10%**) | Worth ~0% at temperature 0, which is the only way vendors benchmark. It only pays when the target actually samples |

Four things that will bite you first — details below: `enable_thinking: false`
**degrades multi-step arithmetic** (use `reasoning_effort: low`); `test_chat.py
--no-think` is a **no-op** on this family; `--gpu-memory-utilization 0.90`
**OOMs** every DSpark config here; and `probabilistic` + `--async-scheduling`
**hangs the engine with no error**.

---

## Verified architecture (`Qwen/Qwen3.8-27B/config.json`, 2026-08-14)

| Property | Value | Consequence |
|---|---|---|
| `architectures` | `Qwen3_5ForConditionalGeneration` | Qwen reused the **Qwen3.5** arch classes for 3.8 |
| `model_type` | `qwen3_5` (text: `qwen3_5_text`) | same family class as Qwen3.5-27B and Qwen3.6-27B |
| `vision_config` | **present** (27-layer, hidden 1152, patch 16) | genuinely multimodal → `--language-model-only` is valid for text-only serving |
| `full_attention_interval` | 4 | 64 layers = 48 Gated DeltaNet + 16 full attention |
| `num_hidden_layers` | 64 | — |
| `hidden_size` / `head_dim` | 5120 / 256 | — |
| `num_attention_heads` / `num_key_value_heads` | 24 / 4 | GQA 6:1 |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 48 | DeltaNet head counts |
| `mtp_num_hidden_layers` | 1 | **MTP head present** — spec decoding available |
| `max_position_embeddings` | 262144 | 256K native context |
| `rope_theta` / `partial_rotary_factor` | 1e7 / 0.25 | partial RoPE, as in 3.6 |
| `mrope_interleaved` / `mrope_section` | true / `[11, 11, 10]` | interleaved multimodal RoPE — **new vs 3.6** |
| `vocab_size` | 248320 | **new tokenizer** — 3.6 was ~152K. Do NOT reuse 3.6 GGUF or jinja assets |

The 2.4T flagship (`Qwen/Qwen3.8-2.4T-A95B`) is the same hybrid design as an MoE
— `Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text`, 92 layers, 512 experts / 10
active, hidden 8192. Far too large for this card; noted only for family context.
**Its chat template differs from the 27B's** — see below.

---

## Controlling thinking

Qwen3.8-27B supports **both** knobs. This is the single most important thing to
get right when repointing clients from 3.6.

### `enable_thinking: false` still works on the 27B

```jsonc
"chat_template_kwargs": {"enable_thinking": false}
```

The 27B template emits a pre-closed `<think>\n\n</think>` in that case, exactly
as 3.6 did, so the Qwen3.6 recipe in `../qwen3.6/README.md` carries over intact.

> **Do not generalize this to the 2.4T flagship.** Its template *raises*
> `'Disabling thinking is not supported.'` on the same input — a 500 from
> template rendering, not a silently-ignored flag. The two models in this family
> genuinely differ here. If you ever point a client at the flagship, that client
> must stop sending `enable_thinking`.

### `reasoning_effort` is the new addition

```jsonc
"chat_template_kwargs": {"reasoning_effort": "low"}   // low | medium | xhigh (default)
```

A **prompt-level** mechanism, not a logits-level one: the template prepends a
sentence to the system message ("Reasoning effort is set to low. Keep your
thinking brief and focused…"). Anything outside those three values raises.
The default is `xhigh` — the most verbose setting — so an un-configured endpoint
is slow and token-hungry out of the box. Setting this is worth it.

### `thinking_token_budget` — hard cap, unverified here

Top-level request field (vLLM ≥0.27), a sampler-level forced `</think>`
independent of the template and gated only on `--reasoning-parser` being set
(all composes here set it). On 3.6 it was exact and cheap. **Not yet re-measured
on 3.8** — re-run the budget sweep from `../qwen3.6/README.md`.

### Other template details

- `add_generation_prompt` emits a trailing `<|im_start|>assistant\n<think>\n`,
  so generation starts **inside** the thinking block with no opening tag in the
  model's own output. Confirm `--reasoning-parser qwen3` copes; if `reasoning`
  comes back empty while `completion_tokens` is high, try `deepseek_r1`.
- `preserve_thinking` (default true) controls whether prior-turn reasoning is
  replayed into context. Setting it false trims history cost on long agent runs.
- Tool calls render as `<tool_call><function=NAME><parameter=ARG>…`, and the
  template injects a strict format reminder into the system prompt. Same XML
  dialect as 3.6 → `--tool-call-parser qwen3_coder` carries over.
- System messages **cannot** contain images or video — the template raises.
- The response field is `message.reasoning`, **not** `reasoning_content` (as in
  3.6). Expect the same here; verify.

---

## Engine support

Both upstreams cut **dedicated `qwen38` image tags on 2026-08-12**, before the
27B was public. Use them; do **not** use `:nightly` or the pip-upgrade-at-boot
entrypoint pattern from the older 3.6 NVFP4 files — that is exactly the drift
hazard this repo has been burned by.

| Engine | Tag | Notes |
|---|---|---|
| vLLM | `vllm/vllm-openai:qwen38-x86_64-cu130` | ✅ verified — 22.9 GB image, reports `v0.1.dev19754+g3a0914114`. Also `qwen38-cu129`, `qwen38-arm64-*` for Spark |
| SGLang | `lmsysorg/sglang:qwen38-27b` | untested here, but **pulled and ready** (17.9 GB, cut 2026-08-14 once the 27B was public). This is the one stack with a published number that could beat ours — see "Still open". The older `:qwen38` tag predates the 27B. |
| llama.cpp | — | GGUF conversion exists (`unsloth/Qwen3.8-27B-GGUF`, full Q2→Q8 ladder), so the new tokenizer is handled upstream. Serving untested. |

Latest stable vLLM release is `v0.27.1` (2026-08-11), which predates the qwen38
tag and almost certainly **cannot** load these checkpoints. Do not "upgrade" to
it. Repin only when a numbered release ships with qwen3.8 support.

Harmless startup noise to expect from the vLLM image — these are `[ERROR]`-level
transformers docstring lint, not failures:

```
[ERROR] `min_frames` is part of Qwen3VLVideoProcessorInitKwargs, but not documented.
[ERROR] `max_frames` is part of Qwen3VLVideoProcessorInitKwargs, but not documented.
```

---

## Measured: FP8 on RTX PRO 6000 (2026-08-14)

`Qwen/Qwen3.8-27B-FP8` via `vllm/vllm-openai:qwen38-x86_64-cu130`.
**Booted first try, no config changes needed** — the qwen38 image handles this
checkpoint natively.

The compose now tracks the [official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
(`--kv-cache-dtype fp8`, no `--language-model-only`, local tuning flags removed).
Two configurations were measured; **they differ on two axes at once**, so the
throughput gap between them is not attributable to either change alone:

| | first boot | recipe-aligned (current) |
|---|---|---|
| KV cache dtype | auto (BF16) | **fp8** |
| Vision tower | disabled (`--language-model-only`) | **loaded** |
| Weights on GPU | 27.57 GiB | 28.43 GiB |
| **KV pool** | 851,007 tokens (3.25x concurrency) | **1,672,297 tokens (6.38x)** |
| `test_chat.py` avg | 49.0 tok/s | 45.6 tok/s |
| `test_tools.py` | 4/4 | 4/4 |

fp8 KV essentially **doubles** the KV pool for ~7% throughput on this workload —
worth it for concurrency or long context, and it is what Qwen and vLLM ship as
the default.

Startup, cold HF cache (first boot). A warm-cache restart is ~200 s total:

| Phase | Time |
|---|---|
| Weight download + load | 447 s |
| Engine init (profile, KV cache, warmup) | 139 s (62.7 s of it torch.compile) |
| CUDA graph capture | 6 s (0.82 GiB) |
| **Total to healthy** | **~10 min** |

Engine picked `CutlassFp8BlockScaledMMKernel`, `Qwen3.5 GDN decode kernel:
triton`, and FlashAttention 2 for the 16 full-attention layers.

### fp8 KV cache is safe here — verified, unlike on 3.6

Every compose in `../qwen3.6/` forbids `--kv-cache-dtype fp8_e4m3` because it
silently corrupted output where DeltaNet met a quantized KV cache
(vllm-project/vllm#26646). **The 3.8 recipe ships fp8 KV as the default**, so
that rule is reversed here — but since the failure mode is wrong answers rather
than a crash, it was verified rather than assumed:

| Probe | Result |
|---|---|
| Needle-in-haystack, ~6.5K tokens, middle | ✅ exact |
| Needle-in-haystack, ~25.9K tokens, middle | ✅ exact |
| Needle-in-haystack, ~51.7K tokens, **early** | ✅ exact |
| Needle-in-haystack, ~51.7K tokens, **late** | ✅ exact |
| Factual recall, powers, prime sequence | ✅ 4/4 |
| `test_tools.py` incl. parallel calls | ✅ 4/4 |

Retrieval is exact at 51.7K tokens at both shallow and deep needle placement —
that is the KV cache being read correctly. Not yet probed near the 262K ceiling.

### ⚠️ Disabling thinking costs arithmetic accuracy

Deterministic, reproducible, and **a regression from 3.6**, where thinking-off
stayed 5/5 on the math set:

| Question | `enable_thinking: false` | `reasoning_effort: low` |
|---|---|---|
| Sum of divisors of 28 (correct: 56) | **120** — 3/3 wrong | **56** — 3/3 right |

Even the cheapest thinking setting fixes it, and `low` costs far less than the
`xhigh` default. **Prefer `reasoning_effort: low` over `enable_thinking: false`**
for anything numeric; reserve the hard off-switch for pure formatting or
extraction work. (Simpler arithmetic — `17*23`, `2^16` — is unaffected.)

### Vision: enabled and verified ✅

Dropping `--language-model-only` (per the recipe) turns on the 27-layer vision
tower. Confirm it is active by the **absence** of this line at startup:

```
INFO ... [registry.py:141] All limits of multimodal modalities supported by the
model are set to 0, running in text-only mode.
```

All probes below ran with `enable_thinking: false` — vision does not require
thinking to work:

| Probe | Image | Result | Prompt tokens | Latency |
|---|---|---|---|---|
| Object ID | Terminator T-800, 110 KB JPEG | ✅ identified the endoskeleton, red eyes, metal skull | 512 | 2.0 s |
| Large image | 2912×1632 RGBA PNG | ✅ read the scene correctly (neon crucifix on a circuit wall) | 4,665 | 4.0 s |
| **Multi-image** | two 1024×1024 PNGs in one request | ✅ described each separately and compared them | 2,082 | 8.6 s |
| **OCR** | synthetic invoice | ✅ **exact** transcription — invoice no., date, all 3 line items, total | 392 | 2.3 s |
| **OCR + reasoning** | same invoice | ✅ all 3 products right, sum 1,498.65 right, caught the deliberately-wrong printed total and computed the 9.60 delta | 407 | 9.7 s |

The last one is the strongest result: it re-derived the arithmetic from OCR'd
values rather than anchoring on the printed TOTAL, with thinking off.

Image cost scales as expected — a 2912×1632 image is ~4.7K prompt tokens, so
budget context accordingly. **Video** is supported by the architecture
(`video_token_id`, `video_preprocessor_config.json`) but is untested here; the
card notes full-frame decoding needs
`--media-io-kwargs '{"video": {"num_frames": -1}}'`, which no compose sets.

### Throughput — the full ladder, fastest first

All on the same card, same image, one model resident at a time.
`benchmarks/RESULTS.log` has every run including the failures.

The headline 111.8 is the conservative number: it is the 7-run `final.sh`
measurement. A 5-run confirmation against the live production endpoint after
deployment gave **114.0**. Treat cross-boot spread on this card as ~3% and do
not read differences smaller than that as real — several apparent orderings in
this table are inside it, which is exactly why the table carries two columns and
why the block-size decision below did not follow the fastest single number.

**Read the two columns as two different questions**, because for speculative
configs they genuinely disagree:

- **sampled** — `../shared/test_chat.py --runs 7 --warmup`. The model's own
  `generation_config` (temperature 1.0, top_p 0.95, top_k 20), thinking on.
  *How fast is our endpoint, for our clients.* This column sorts the table.
- **greedy** — `./bench_greedy.py --runs 5`. temperature 0, thinking off,
  `ignore_eos`, 256 tokens. *The protocol every vendor checkpoint card uses*,
  reproduced so their numbers and ours are comparable.

| # | Config | sampled | greedy | vs FP8 |
|---|---|---|---|---|
| **1** | **W4A4 NVFP4 + DSpark-7 + `probabilistic`** ← fastest single stream **only** | **111.8** | 112.2 | **2.45x** |
| 2 | W4A4 NVFP4 + DSpark-5 + `probabilistic` | 111.1 | 107.0 | 2.44x |
| 3 | W4A4 NVFP4 + DSpark-7 + `probabilistic` + `--max-num-batched-tokens 16384` | 110.0 | 110.6 | 2.41x |
| 4 | W4A4 NVFP4 + DSpark-6 + `--async-scheduling` | 109.5 | 114.0 | 2.40x |
| **5** | **W4A4 NVFP4 + MTP-2 + `probabilistic`** ← **PRODUCTION** (wins from concurrency 8 up) | 108.7 | **122.1** | 2.38x |
| 6 | W4A4 NVFP4 + DSpark-7 + `dspark_draft_topk: 8` | 107.6 | 109.3 | 2.36x |
| 7 | W4A4 NVFP4 + DSpark-5 + `--async-scheduling` | 103.8 | **116.9** | 2.28x |
| 8 | W4A4 NVFP4 + DSpark-4 + `--async-scheduling` | 102.8 | 111.4 | 2.25x |
| 9 | W4A4 NVFP4 + DSpark-7 + `--async-scheduling` | 102.0 | 112.3 | 2.24x |
| 10 | W4A4 NVFP4 + DSpark-7 (no extra flags) | 101.6 | — | 2.23x |
| 11 | Unsloth NVFP4 + MTP-2 *(the 2026-08-15 champion)* | 97.9 | — | 2.15x |
| 12 | W4A4 NVFP4 + DSpark-3 + `--async-scheduling` | 96.4 | 104.5 | 2.11x |
| 13 | W4A4 NVFP4 + MTP-2 | 95.9 | — | 2.10x |
| 14 | **W4A4 NVFP4, no speculation** | 77.1 | 78.8 | 1.69x |
| 15 | Unsloth NVFP4, no speculation | 58.7 | — | 1.29x |
| 16 | FP8, official recipe | 45.6 | — | 1.00x |

**Row 14 is the load-bearing one.** With speculation off, all three benchmark
protocols agree to within 2% (77.1 sampled / 77.3 fixed-length / 78.8 greedy).
Every disagreement elsewhere in the table is therefore *acceptance-rate
sensitivity*, not benchmark noise — which is also why a config can win one
column and lose the other.

#### Why W4A4 beats "NVFP4" (rows 14 vs 15, +31%)

Both checkpoints are advertised as NVFP4. They are not the same thing:

| | `unsloth/…-NVFP4` | `gittensor-model-hub/…-NVFP4-RTX5090` |
|---|---|---|
| Producer | compressed-tensors, `MIXED_PRECISION` | modelopt, `NVFP4` |
| Weights | 4-bit on `mlp.{gate,up,down}_proj` **only** | 4-bit on `targets: ["Linear"]` |
| **Activations** | **16-bit** | **4-bit** |
| Left higher | attention, DeltaNet gates, **layers 56-63 MLPs** (FP8) | only lm_head, embeddings, DeltaNet conv1d/in_proj_a/b, vision tower, MTP head |
| Size | 23.44 GB | 20.62 GB |
| **tok/s** | **58.7** | **77.1** |

On sm_120 the FP4 tensor cores only engage when the **activations** are FP4 too.
A W4A16 checkpoint dequantizes and runs the GEMM at 16-bit, so it buys VRAM and
nothing else. Confirm which path you got — the log line you want is

```
Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM
```

and `quantization=modelopt_fp4` in the engine config dump. Note vLLM
**auto-detects** this correctly; passing `--quantization modelopt` is not needed
despite what the checkpoint card says.

This is also why the AWQ-INT4 recipes circulating for **3090s** must not be
copied here: INT4 + Marlin is an Ampere workaround for a card with no FP4/FP8
path at all, and it is W4A16 — the slow column above.

#### Why the built-in MTP head lost to DSpark on single stream (rows 13 vs 9)

Counter-intuitively, MTP got **worse** as the target got faster:

| Target | base tok/s | MTP-2 tok/s | mean accept | avg draft acceptance |
|---|---|---|---|---|
| Unsloth NVFP4 (W4A16) | 58.7 | 97.9 (**1.67x**) | 2.19 | 59.4% |
| W4A4 NVFP4 | 77.1 | 95.9 (**1.24x**) | ~1.9 | 38-55% |

The MTP head ships in BF16 in both checkpoints. The harder the target is
quantized, the further its output distribution moves from what that BF16 head
predicts, so acceptance falls. A faster target with a worse drafter is close to
a wash — and it means **MTP depth tuned on one checkpoint does not transfer to
another**, even at the same bit-width label.

`RadixArk/Qwen3.8-27B-DSpark` is a *separately trained* 1.36B block drafter
(5 layers, EAGLE-style Markov + confidence heads) rather than a single reused
layer, so it does not inherit that problem.

⚠️ **This conclusion reverses under concurrency.** MTP loses to DSpark at one
request in flight and beats it decisively at eight or more, because MTP drafts 2
tokens per step where DSpark drafts 7 and the wasted draft compute is what
dominates once the GPU is compute-bound. See
[Production selection](#production-selection-concurrency-changes-the-answer-2026-08-18).

#### `draft_sample_method: probabilistic` — the flag nobody recommends (+10%)

vLLM's default is `greedy`: it drafts the drafter's argmax. That is the correct
proposal distribution only if the **target** is also greedy. Our endpoint serves
Qwen's own sampling (temperature 1.0), so the target samples from a 248K-way
distribution and rejects argmax drafts hard.

| | sampled (temp 1.0) | greedy (temp 0) |
|---|---|---|
| `draft_sample_method: greedy` (default) | 102.0 | 112.3 |
| `draft_sample_method: probabilistic` | **111.8** | 112.2 |

**+10% where it matters, 0% at temperature 0.** Which is precisely why no vendor
card mentions it: they all benchmark greedy, where the flag is invisible.

#### DSpark block size: the published 7 is not the optimum — but ship it anyway

vLLM enforces `num_speculative_tokens >= block_size` from the drafter's
`config.json`, so sweeping below 7 needs a locally edited copy of the drafter
(recipe below). On the prose benchmark prompt:

| block | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| greedy tok/s | 104.5 | 111.4 | **116.9** | 114.0 | 112.3 |

Peak at 5, worth ~4%. **This directory still ships 7**, for two reasons.

First, block 5 was less reproducible across boots (116.9 / 115.8 / 113.9 / 113.7
for one config) while block 7 was rock-solid (112.3 / 112.4 / 112.3 / 112.2).

Second and decisive: **acceptance is workload-dependent, and the prose prompt is
speculation's worst case.** Same running server, same block 7:

| Workload | mean accept | per-position acceptance |
|---|---|---|
| Prose (the benchmark prompt) | 2.2 | 0.58, 0.37, 0.23, 0.15, 0.08, 0.04, 0.02 |
| Structured / list output | **4.86** | 0.83, 0.69, 0.61, 0.54, 0.47, 0.40, **0.32** |

On structured and code-like output the 7th draft slot still lands **32%** of the
time. Truncating the block to 5 to win 4% on prose would give that up. Keep 7,
and keep the unmodified upstream checkpoint.

To reproduce the short-block variants anyway:

```bash
# copy the snapshot, hardlink the weights, edit ONE field
mkdir -p drafters/dspark-block5
docker run --rm -v qwen38_huggingface_cache:/c -v "$PWD/drafters":/out alpine sh -c \
  'S=$(ls -d /c/hub/models--Doopeworld--Qwen3.8-27B-DSpark-vLLM/snapshots/*/ | head -1);
   cp -L $S* /out/dspark-block5/'
python3 -c "import json;p='drafters/dspark-block5/config.json';c=json.load(open(p));\
c['block_size']=5;c['dspark_block_size']=5;json.dump(c,open(p,'w'),indent=1)"
# then mount it:  -v "$PWD/drafters":/drafters:ro
#                 "model":"/drafters/dspark-block5","num_speculative_tokens":5
```

`drafters/` is gitignored — regenerate, do not commit 2.5 GB of weights.

#### Everything that did NOT work

Recorded because each cost a boot cycle, and because "we tried it" is worth as
much as "it worked":

| Attempt | Result |
|---|---|
| `--async-scheduling` **+** `probabilistic` | **hangs the engine, no error** — see traps |
| `--async-scheduling` alone | +10% (102 -> 112 greedy). Real, but not combinable with the above, and worth about the same |
| `VLLM_USE_V2_MODEL_RUNNER=1` | no effect (107.5 vs 109.4). Aimed at GB200/MoE |
| `VLLM_QWEN3_5_GDN_DECODE_KERNEL=fused` | **refuses to load**: *"fused requires a BF16 Qwen3.5 model … and an SM100 GPU"*. SM100 is B200/B300; this card is SM120, and the model is quantized. Same "Blackwell ≠ Blackwell" trap as `trtllm_mha` |
| `dspark_draft_topk: 8` | **hurts** (112.4 -> 109.3). It cheapens the Markov projection but costs acceptance (2.24 -> 2.11) |
| `--max-num-batched-tokens 16384` | wash (110.0 vs 111.8). vLLM warns that speculation clamps `max_num_scheduled_tokens` to 2048 — that warning is about prefill at concurrency, not single-stream decode |
| `--gpu-memory-utilization` 0.90 / 0.92 / 0.95 | 0.90 and 0.95 **OOM**; 0.92 boots but is no faster. Utilisation does not change decode rate (block-7: 112.3 at 0.90 vs 112.4 at 0.88) |
| `gittensor-model-hub/…-DSpark-NVFP4` drafter | **cannot load in vLLM** — see traps |

#### Arithmetic that explains the shape of all of this

Decode here is **entirely memory-bound**, and that is why speculation pays at all:

```
weights read per forward   17.5 GB / 1.79 TB/s  =  9.8 ms   (independent of batch)
compute for 16 tokens      2 x 27e9 x 16        =  0.6 ms   at ~1.5 PFLOPS FP4
```

Verifying 16 draft tokens costs the same as verifying 1. So the *verify* side of
speculation is free, and every bit of speculation overhead lives in the
**drafter**. Measured: 13 ms/step without speculation, 22.5 ms/step with it —
9.5 ms of drafting per step, which is why `dspark_draft_topk` looked promising
and why the block-size curve has an interior optimum at all.

### Correctness under speculation — verified, and it was an open item

The 2026-08-15 README listed "correctness under MTP" as the highest-value gap:
throughput had been swept, correctness had not, and on 3.6 speculation crashed
the engine on mixed-modality batches. Re-run against the production DSpark
config (`docker-compose.vllm-27b-nvfp4-gittensor-dspark-rtx.yml`):

| Probe | Result |
|---|---|
| `test_tools.py` incl. **parallel** calls | ✅ **4/4** |
| Vision, 0.5 MP JPEG | ✅ identified the T-800 endoskeleton (506 prompt tokens) |
| Vision, 1.05 MP PNG | ✅ correct scene (1,046 prompt tokens) |
| Vision, **4.75 MP** PNG | ✅ correct scene (**4,663** prompt tokens) |
| Needle in haystack, **43,247 tokens**, needle at 85% depth | ✅ exact recall |
| Arithmetic (`reasoning_effort: low`), 3 questions | ✅ 3/3 incl. divisors-of-28 |
| **Determinism at temperature 0**, 3 identical requests | ✅ **byte-identical** |

The last row is the one that matters for a speculative config: rejection
sampling is supposed to be *lossless*, so a correctly implemented drafter cannot
change the output distribution. It doesn't. Vision also survives — no
mixed-modality crash, though note the startup warning that the drafter gets
**text-only** inputs (`Draft model Qwen3DSparkForCausalLM does not support
external multimodal embeddings`), so acceptance on image-conditioned turns is
lower than on text.

The 4.75 MP image doubles as the mandated regression test for the Unsloth
2048-token tokenizer trap documented below: 4,663 prompt tokens is far past 2048.

### Four traps found while chasing this, each of which looks like something else

**1. A drafter silently inherits the target's quantization.**
`gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4` will not load in vLLM. It dies with

```
RuntimeError: The size of tensor a (128) must match the size of tensor b (256)
              at non-singleton dimension 1
```

which names no parameter and reads like a corrupt download. It is neither. The
parameter is `markov_head.markov_w2.weight`, a `ParallelLMHead` of width
`markov_rank=256`; allocating it as NVFP4 packs two values per byte, giving
width 128, so the real 256-wide BF16 tensor cannot be copied in. The drafter's
own `ignore` list *does* exempt the Markov head — but it spells the entries as
`"markov_head*"`, SGLang glob syntax, and vLLM's `check_equal_or_regex_match()`
accepts only exact layer names or an explicit `re:` prefix. So the entry matches
nothing, the head is quantized against the author's intent, and the failure
surfaces as a shape error 200 lines later.

To get the parameter name yourself rather than guessing, wrap the loader:

```python
# vllm/model_executor/models/utils.py, in AutoWeightsLoader._load_param
try:
    weight_loader(param, weight_data)
except Exception as _e:
    raise RuntimeError(f"name={weight_qualname!r} param={tuple(param.shape)} "
                       f"loaded={tuple(weight_data.shape)}") from _e
```

**2. `draft_sample_method: probabilistic` + `--async-scheduling` hangs the engine.**
Not a crash — worse. The API server answers, the model loads, 90 GB is
allocated, the first request is accepted... and GPU utilisation sits at **0%**
while `/health` never returns again. There is no error in the log. A harness
that waits for the container to exit or for a benchmark to return will hang with
it; every benchmark call in `sweep.sh` is wrapped in `timeout` for exactly this
reason. `probabilistic` is what the drafter's own card recommends, so this
combination is easy to walk into. Not yet isolated to one of the two flags.

**3. Vendor tok/s and our tok/s are different experiments.**
Checkpoint cards quote `temperature=0` with thinking off. This repo's standard
`test_chat.py` run uses the model's own `generation_config`
(`temperature=1.0`, `top_p=0.95`, `top_k=20`) with thinking on. Speculative
acceptance is a function of the target's sampling distribution — at temperature
0 a draft token is accepted iff it is the argmax, at temperature 1.0 the target
samples from a 248K-way distribution and rejects far more — so the two protocols
are not comparable and the gap is *specific to speculative configs*. Hence
`bench_greedy.py`: it reproduces the vendor protocol so the comparison is
honest, and both numbers are reported for every config below.

**4. Counting stream chunks undercounts throughput by ~2x — but only when
speculation works.**
Under speculative decoding vLLM emits every token accepted in one verify step as
a **single** SSE chunk, so chunks number roughly `tokens / acceptance_length`. A
benchmark that counts chunks therefore reports ~half the real rate, and the
error scales with how *well* speculation is working — the fastest configs look
the slowest. Always read `usage.completion_tokens` from the final chunk
(`stream_options: {"include_usage": true}`). `../shared/test_chat.py` does this
correctly; `bench_greedy.py` did not on its first draft, and reported 51.5 tok/s
for a config actually running at 112.3.

### `test_chat.py --no-think` is a NO-OP here — do not report its numbers

The flag sends `enable_thinking` as a **top-level** request field
(`test_chat.py:48`). vLLM ignores that; the switch only works inside
`chat_template_kwargs`. Measured on the same prompt ("What is 17*23?"):

| Request form | reasoning | completion tokens | answer |
|---|---|---|---|
| top-level `enable_thinking: false` (what `--no-think` sends) | 89 chars | 36 | 391 ✓ |
| `chat_template_kwargs: {"enable_thinking": false}` | 0 chars | **4** | 391 ✓ |

Same answer, 9x the tokens. This matches the Qwen3.6 finding — the flag has
never worked on this family under vLLM.

### Tool calling: 4/4

`test_tools.py` passes clean with `--tool-call-parser qwen3_coder`, including
**parallel** tool calls (three dispatched in one turn, all three results folded
into the final answer). The `<tool_call><function=…><parameter=…>` XML dialect
is unchanged from 3.6, so agent configs that already speak it need no change.
`message.reasoning` is populated as expected — not `reasoning_content`.

---

## Production selection: concurrency changes the answer (2026-08-18)

Everything above measures **one request at a time**. That ranking does not
survive contact with a real endpoint. Measured with GuideLLM 0.5.4
(`--profile concurrent`, chat profile: ~2000 in / ~300 out) plus a custom
mixed text+image load generator, all on the same card:

### Aggregate output tok/s vs concurrency — chat

| Config | 1 | 2 | 4 | 8 | 15 | ~30 | ~55 |
|---|---|---|---|---|---|---|---|
| **W4A4 + MTP-2 + `probabilistic`** ← **production** | 111.3 | 220.5 | 388.9 | **639.5** | **913.3** | **1158.6** | 1213.6 |
| W4A4 + MTP-1 | 96.7 | 178.3 | 331.0 | 536.5 | 838.6 | 1089.1 | **1233.6** |
| W4A4 + MTP-2 | 100.3 | 202.4 | 367.5 | 599.3 | 835.3 | 1090.8 | 1163.8 |
| W4A4 + DSpark-3 | 104.1 | 213.2 | 375.0 | 607.3 | 807.3 | 1059.0 | 1131.2 |
| W4A4, **no speculation** | 80.6 | 137.8 | 245.4 | 446.2 | 707.7 | 1004.8 | 1136.2 |
| W4A4 + DSpark-7 + `probabilistic` + `--max-num-batched-tokens 16384` | **130.3** | **233.3** | **414.2** | 605.5 | 774.3 | 932.7 | 857.2 |
| W4A4 + DSpark-7 + `probabilistic` *(the single-stream champion)* | 127.2 | 231.8 | 401.3 | 584.8 | 768.6 | 891.3 | 867.1 |
| W4A4 + DSpark-7 + adaptive schedule `[[1,16,7],[17,2048,0]]` | 125.4 | 229.2 | 404.0 | 590.7 | 781.3 | 817.3 | 811.0 |

**The single-stream winner is the worst config in the table above concurrency 8.**
DSpark-7 leads at 1–4 concurrent and is dead last by 30. The config that wins in
production ranked **12th** on the single-stream table.

### Why: speculation is free when memory-bound and expensive when compute-bound

At concurrency 1 decode is memory-bound — the arithmetic earlier in this README
shows verifying 16 draft tokens costs the same as verifying 1, so a long draft
block is nearly free and DSpark's block of 7 wins. As the batch fills, the GPU
becomes **compute**-bound and every drafted-but-rejected token is real stolen
work. Draft cost per request per step is the whole story:

| Config | draft slots/request/step | tok/s @1 | tok/s @30 |
|---|---|---|---|
| DSpark-7 | 7 | 127.2 | 891.3 |
| DSpark-3 | 3 | 104.1 | 1059.0 |
| MTP-2 | 2 | 100.3 | 1090.8 |
| MTP-1 | 1 | 96.7 | 1089.1 |
| none | 0 | 80.6 | 1004.8 |

Monotonic in both directions, and it crosses over. **The optimal draft length
falls as concurrency rises** — which is also why the single-stream block-size
sweep (optimum 5–7) is the wrong instrument for choosing a production config.

### Two things that make MTP the production choice beyond raw tok/s

**KV pool.** Speculation reserves draft slots out of the KV cache, and DSpark
reserves far more:

| Config | KV pool (tokens) | concurrency @ full 262K context |
|---|---|---|
| no speculation | 1,831,994 | 6.99x |
| MTP-1 | 1,610,313 | 6.14x |
| **MTP-2 + `probabilistic`** | **1,557,969** | **5.94x** |
| DSpark-3 | 1,127,821 | 4.30x |
| DSpark-7 | 967,870 | 3.69x |

**Images.** The DSpark drafter cannot see image embeddings. vLLM says so at
startup and then carries on:

```
Draft model Qwen3DSparkForCausalLM does not support external multimodal
embeddings. Embeddings from the target model will not be passed to the drafter;
using text-only draft inputs instead.
```

So on every image turn DSpark drafts blind and the work is wasted. MTP is part
of the target model and reads its hidden states, so it does not have this
problem. Per-stream tok/s on a 50/50 text+image mix (`mixed_load.py`):

| Config | conc 8 text | conc 8 image | conc 32 text | conc 32 image |
|---|---|---|---|---|
| **MTP-2 + `probabilistic`** | 73.7 | **76.9** | 38.1 | **36.1** |
| MTP-2 | 71.2 | 66.0 | 39.0 | 33.7 |
| no speculation | 58.1 | 61.8 | 37.4 | 36.3 |
| DSpark-3 | 64.9 | 57.5 | 41.8 | 25.4 |
| DSpark-7 + `probabilistic` | 65.9 | 47.5 | 31.8 | **18.4** |

Under DSpark an image request decodes at **half** the rate of a text request in
the same batch. Under MTP the two are within a few percent. For a multimodal
endpoint that is decisive on its own.

**Stability: zero errors everywhere.** Every config in this section completed
every level with no failed requests, no crashes and no mixed-modality faults —
including 128-request mixed text+image bursts at concurrency 32, and images from
0.5 MP to 4.75 MP interleaved with text-only turns. The 3.6-era fear that
speculation crashes on mixed-modality batches does not reproduce on 3.8.

> One retraction worth recording: a first DSpark-3 run appeared to crash under
> load (guidellm "worker process group startup failed", then connection
> refused). It was the **benchmark harness** dying, not the engine. A clean
> re-run passed with `end_status=running` and zero error lines in the server
> log. `prodtest.sh` now always saves the server log, because from the client
> side a dead harness and a dead engine look identical.

### Scenario matrix (production config's predecessor, DSpark-7, concurrency 1→32)

Peak aggregate tok/s and the concurrency it occurs at, showing how strongly the
workload shape matters:

| Scenario | in / out tokens | peak tok/s | at conc | bottleneck |
|---|---|---|---|---|
| codegen | 4K / 1500 | **994.5** | 20 | decode |
| chat | 2K / 300 | 891.3 | 29 | decode |
| agentic | 16K / 800 | 370.5 | 11 | prefill + KV |
| rag | 8K / 256 | 326.1 | 14 | prefill |
| summarization | 12K / 300 | 251.4 | 22 | prefill |

Long-prompt/short-output work is prefill-bound and peaks 3–4x lower than
decode-heavy work. Speculation does nothing for prefill, which is a second
reason its advantage shrinks on rag/summarization endpoints.

### Practical capacity on this one card

Sweet spot = request latency p50 under ~5 s and TPOT under ~15 ms, on the
production config:

| Scenario | sweet-spot conc | tok/s there | lat p50 | practical bursty users |
|---|---|---|---|---|
| chat | 8 | 639 | 3.8 s | ~12 |
| codegen | ~8 | ~600 | — | ~8 (decode-bound) |
| rag | 4 | ~230 | 4.3 s | ~5 |
| summarization | 4 | ~195 | 6.6 s | ~4 |
| agentic | 4 | ~247 | 12.6 s | 1–2 |

> **One RTX PRO 6000 serves roughly 12 concurrent chat users, ~5 RAG users, or
> 1–2 agentic agents at interactive latency — and up to ~1,200 tok/s aggregate
> if you are willing to accept 16 s latency at concurrency 55.**

### Which config should you actually run

| If your endpoint is… | Use | Why |
|---|---|---|
| **Multi-user / multimodal / agentic (the default)** | `docker-compose.vllm-27b-nvfp4-gittensor-mtp-prob-rtx.yml` | Best from concurrency 8 up, +61% KV pool, images don't pay double |
| Single user, latency above all (one dev, one IDE) | `docker-compose.vllm-27b-nvfp4-gittensor-dspark-rtx.yml` | +14% at concurrency 1 (127 vs 111), and nothing else is in flight to lose |
| Batch/offline throughput, no latency target | MTP-1 | Highest measured aggregate, 1233.6 tok/s at concurrency 55 |

### Benchmarking notes, so these numbers can be reproduced or challenged

- **GuideLLM's TTFT is unusable against vLLM.** Its median reads ~0 ms because
  vLLM emits an SSE frame carrying an empty role delta the instant a request is
  admitted, and guidellm timestamps that as the first token. Judge
  responsiveness by request latency instead. `summarize_gllm.py` prints the mean
  and flags this.
- **Never pipe guidellm through `head`/`tail`/`tee`** — it SIGPIPE-deadlocks.
  `gllm.sh` redirects to a file and parses the JSON afterwards.
- The concurrency column is guidellm's *achieved* mean concurrency, so requested
  32 shows up as ~29–30 and 64 as ~51–55.
- `mixed_load.py` exists because guidellm is text-only and cannot construct the
  batch that actually matters here: text-only and image-bearing requests
  in flight simultaneously.


---

## Scaling grid: prompt length x concurrency (2026-08-22)

The section above sweeps concurrency at one prompt shape. This one sweeps
**prompt length as well**, because that is the axis that decides where the card
runs out of breath — and it moves the answer by more than 5x. Production config
(W4A4 NVFP4 + MTP-2 + `probabilistic`, fp8 KV, `--max-model-len 262144`,
util 0.88), one boot, GuideLLM 0.5.4, output pinned to 300 tokens (132-500) so
prompt length is the only variable. Zero errored requests anywhere in this grid.

### Aggregate output tok/s

| PP \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| **512** | 145.2 | 287.1 | 539.6 | 949.2 | 1454.7 | 1880.8 | **2375.6** |
| **2K** | 129.5 | 239.1 | 422.5 | 690.0 | 955.8 | 1199.8 | **1270.2** |
| **4K** | 121.6 | 214.5 | 356.0 | 524.1 | 661.9 | **765.6** | 752.6 |
| **8K** | 106.1 | 175.8 | 261.5 | 334.2 | 391.7 | 409.0 | **423.7** |
| **10K** | 101.5 | 167.4 | 236.5 | 284.3 | 311.9 | **333.5** | 324.2 |

**The card's peak aggregate throughput varies 5.6x with prompt length alone**
(2375.6 at 512 vs 423.7 at 10K), on identical hardware, weights and flags. Every
"tokens per second" number for this GPU is meaningless without the prompt length
attached.

Nothing in this grid actually turns over inside the tested range — the curves
*flatten*, and they flatten earlier the longer the prompt: 512 and 2K are still
climbing at 64, 4K/8K/10K are flat from 32 on. Decode-bound work keeps rewarding
concurrency; prefill-bound work stops.

### Per-stream output tok/s — what one user feels

| PP \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| **512** | 143.5 | 150.2 | 138.0 | 122.6 | 93.2 | 65.4 | 37.6 |
| **2K** | 119.3 | 114.4 | 105.3 | 85.2 | 58.6 | 36.7 | 19.0 |
| **4K** | 114.7 | 107.0 | 87.0 | 65.1 | 40.4 | 22.9 | 9.9 |
| **8K** | 100.7 | 87.4 | 66.3 | 41.6 | 22.0 | 13.1 | 6.5 |
| **10K** | 96.7 | 83.4 | 56.9 | 35.6 | 18.1 | 10.4 | 5.1 |

### Request latency p50 (s)

| PP \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| **512** | 2.1 | 2.1 | 2.3 | 2.5 | 3.3 | 4.7 | 8.1 |
| **2K** | 2.2 | 2.6 | 2.8 | 3.5 | 5.0 | 8.4 | 15.5 |
| **4K** | 2.5 | 2.7 | 3.6 | 4.7 | 7.3 | 13.2 | 28.2 |
| **8K** | 2.9 | 3.4 | 4.6 | 7.1 | 12.3 | 23.0 | 46.0 |
| **10K** | 3.0 | 3.8 | 5.5 | 8.1 | 14.6 | 29.1 | 60.2 |

### Long context: 100K and 150K

The interesting question is not whether the KV pool holds N long-context streams
— it does — but whether running them concurrently buys anything. It does not.

| PP | conc | total tok/s (prompt+output) | output tok/s | per-stream tok/s | lat p50 | req/min |
|---|---|---|---|---|---|---|
| **100K** | 1 | 5903 | 16.1 | 17.0 | 17.8 s | **3.97** |
| 100K | 4 | 6466 | 19.9 | 4.5 | 65.1 s | 3.81 |
| 100K | 8 | 6609 | 21.1 | 2.4 | 137.9 s | 3.17 |
| 100K | 16 (12.2 achieved) | 6049 | 17.4 | 1.2 | 263.5 s | 3.17 |
| **150K** | 1 | 5041 | 9.7 | 10.1 | 32.6 s | **2.06** |
| 150K | 4 | 5300 | 11.3 | 2.2 | 128.2 s | 1.75 |
| 150K | 8 | 5519 | 12.5 | 1.1 | 273.6 s | 1.27 |
| 150K | 10 (5.6 achieved) | 5533 | 10.8 | 1.1 | 271.3 s | 1.27 |

**At 100K the card is prefill-saturated at ~6,000-6,600 total tok/s and at 150K
at ~5,000-5,500, and concurrency does not move that number.** Going from 1 to 12
concurrent 100K streams changes total throughput by less than 12% while
multiplying request latency by 15x (17.8 s -> 263.5 s) — and *completed requests
per minute goes down*, from 3.97 to 3.17. The queue is not buying throughput, it
is only buying waiting.

So the honest capacity statement for long context on one RTX PRO 6000 is a
**rate, not a concurrency**: ~4 requests/min at 100K, ~2 requests/min at 150K.
Read `output tok/s` here with care — at 100K prompts the 300 generated tokens are
0.3% of the work, which is why that column reads 16-21 while the card is doing
6,000+.

KV is not the binding constraint at these lengths. The pool is 1,605,632 tokens,
so 16 x 100K (1.6M) just fits and 10 x 150K (1.5M) fits — throughput saturates
long before the pool does. That reverses the intuition from the 2K-prompt
section, where the limit is compute and 55 streams use only ~127K of pool.

### Two measurement traps this grid walked into

Both produced plausible-looking numbers that were wrong, and both are
client-side:

1. **GuideLLM's request timeout is 60 s by default and an aborted request is
   booked as an ERROR, not as slow.** At 100K the first run returned 33 errored
   / 4 ok at conc 16 and a latency curve that looked like an engine failure. The
   engine was fine — every request simply took longer than the harness allowed.
   `gllm.sh` now exposes `GLLM_TIMEOUT` (`--backend-kwargs '{"timeout": N}'`) and
   the long-context levels run at 1800.

2. **A 60 s window is not a measurement at high concurrency with long prompts.**
   At 10K x 64 vLLM logs `Running: 6 reqs, Waiting: 58` — the window closes while
   the engine is still draining the prefill backlog, and because guidellm
   averages only *completed* requests the level reads far too low. 10K x 64
   measured **166.9** tok/s at 60 s and **324.2** at 300 s. The 8K and 10K cells
   at conc 32/64 in the tables above are the 300 s numbers; everything else is
   60 s, which was verified to be enough (those levels show `Waiting: 0`).

   The apparent "collapse" of the longest-prompt curve at conc 64 was entirely
   this artifact. There is no collapse — there is a plateau.

3. **`prompt_tokens_stdev=0` is rejected by guidellm's pydantic config**
   (`Input should be greater than 0`), which kills the run at data-deserialization
   before a single request is sent. Fixed prompt lengths have to be expressed as
   a tight band (`stdev=1`, min/max +-1%).

### Reproducing

```bash
./scaling.sh nvfp4-w4a4 gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090  # PP grid
./lc_rerun.sh          # 100K / 150K against an already-running container
./recheck_run.sh       # 300s steady-state cells for 8K/10K at conc 32-64
python3 scaling_table.py > benchmarks/SCALING-GRID.md
```

Prefix caching is on (vLLM default) but **hit rate was 0.0% throughout** —
guidellm's synthetic prompts are unique, so the prompt-length axis measures real
prefill and not cache hits.

> ⚠️ Do not splice rows from this grid against the 2026-08-18 concurrency table.
> Same image (`v0.1.dev19754+g3a0914114`), same flags, same checkpoint — but this
> grid's 2K row reads 129.5 tok/s at conc 1 where the 08-18 chat row reads 111.3,
> on a slightly narrower prompt distribution. MTP acceptance is prompt-sensitive
> and that is more than the ~3% cross-boot spread. Each table is internally
> consistent; across tables, only compare shapes, not absolute numbers.

### Not covered

**FP8 under concurrency is still not measured** — this grid is 4-bit only, by
choice. The single-stream FP8 datapoint (45.6 tok/s, 2.45x slower than W4A4)
remains the only fp8 number here, so no fp4-vs-fp8 scaling comparison exists.
The `scaling.sh` harness takes a checkpoint id as its second argument and the
FP8 checkpoint ships `mtp.*` weights, so the identical grid can be run against
`Qwen/Qwen3.8-27B-FP8` whenever it is wanted.


---

## Ports

Fresh block — nothing in this repo used 11484+ before.

| Port | File |
|---|---|
| 11484 | vLLM 27B: BF16 / FP8 / FP8+MTP (one at a time) |
| 11485 | SGLang 27B |
| 11486 | llama.cpp 27B |
All vLLM variants share **11484** and the served name `qwen3.8-27b`, so any of
them is a drop-in swap for an existing harness. Only one can run at a time on
this card — stop the other first.

Distinct ports for the three engines so a 3-way runtime A/B can run
back-to-back without editing configs, matching the
`../qwen3.6/benchmarks/v0230_3way_results.txt` setup.

---

## Files

Sorted the same way as the throughput table: fastest first.

| File | Checkpoint | Status |
|---|---|---|
| `docker-compose.vllm-27b-nvfp4-gittensor-mtp-prob-rtx.yml` | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (**community**) | ✅ **VERIFIED — PRODUCTION.** 108.7 tok/s single stream (122.1 greedy, best here) but **1,159 at concurrency 30**, +61% KV pool, images decode at full speed. Tools 4/4, 43K needle, temp-0 deterministic, zero errors under load. |
| `docker-compose.vllm-27b-nvfp4-gittensor-dspark-rtx.yml` | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` + `Doopeworld/Qwen3.8-27B-DSpark-vLLM` (both **community**) | ✅ **VERIFIED — FASTEST SINGLE STREAM, 111.8 tok/s.** Correct choice ONLY for a one-user endpoint: collapses to 891 tok/s at concurrency 30 and halves image decode rate. |
| `docker-compose.vllm-27b-nvfp4-gittensor-mtp-rtx.yml` | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (**community**) | ✅ VERIFIED. 95.9 tok/s single stream (no `probabilistic`). Kept as the controlled A/B for what that flag is worth. |
| `docker-compose.vllm-27b-nvfp4-unsloth-mtp-rtx.yml` | `unsloth/Qwen3.8-27B-NVFP4` (**community**) | ✅ VERIFIED. 97.9 tok/s — the 2026-08-15 champion, now third. Superseded, kept for provenance. |
| `docker-compose.vllm-27b-nvfp4-gittensor-rtx.yml` | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (**community**) | ✅ VERIFIED. 77.1 tok/s no-speculation baseline — the W4A4 reference point, and the row where all three benchmark protocols agree. |
| `docker-compose.vllm-27b-nvfp4-unsloth-rtx.yml` | `unsloth/Qwen3.8-27B-NVFP4` (**community**) | ✅ VERIFIED. 58.7 tok/s. Kept as the W4A16-vs-W4A4 control. |
| `docker-compose.vllm-27b-fp8-rtx.yml` | `Qwen/Qwen3.8-27B-FP8` (first-party) | ✅ VERIFIED. 45.6 tok/s, tracks the official recipe. The **quality/first-party** option, not the fast one. |
| `docker-compose.vllm-27b-bf16-rtx.yml` | `Qwen/Qwen3.8-27B` (first-party) | Unverified. ~54 GB. Only worth it as a quality reference vs FP8. |
| `docker-compose.vllm-27b-fp8-mtp-rtx.yml` | `Qwen/Qwen3.8-27B-FP8` (first-party) | Unverified. Matched drafter/target precision, but on a 45.6 tok/s base — no longer interesting now that DSpark exists. |
| `docker-compose.sglang-27b-rtx.yml` | `Qwen/Qwen3.8-27B` (first-party) | Unverified scaffold. |
| `docker-compose.llama-27b-q4-rtx.yml` | `unsloth/Qwen3.8-27B-GGUF` (**community**) | Unverified scaffold. |

### Tooling added 2026-08-18

| File | What it is |
|---|---|
| `sweep.sh` | Boots one config via `docker run`, waits crash-loop-aware, runs both benchmarks, appends one line to `benchmarks/RESULTS.log`, tears down with SIGTERM + 90 s. Every benchmark call is `timeout`-wrapped. |
| `final.sh` | Same, but 7 sampled + 5 fixed-length + 5 greedy runs. Used to break ties the 3-run sweep could not. |
| `bench_greedy.py` | The vendor-card protocol (temperature 0, thinking off, `ignore_eos`). `--temperature 1.0 --think` gives a fixed-length *sampled* measurement. |
| `run-experiment.sh` | Same idea for a compose file rather than ad-hoc args. |
| `prodtest.sh` | Boot a config, run the guidellm chat concurrency sweep, then mixed text+image load at 8 and 32, save the server log, tear down. |
| `gllm.sh` | One guidellm concurrency sweep for a named scenario (chat/rag/agentic/codegen/summ). Redirects rather than pipes — guidellm SIGPIPE-deadlocks behind `head`/`tail`/`tee`. |
| `summarize_gllm.py` | Flattens `benchmarks.json` into one row per concurrency level. |

### Tooling added 2026-08-22

| File | What it is |
|---|---|
| `scaling.sh` | Boots one checkpoint and runs the whole prompt-length grid (512 -> 10K at concurrency 1-64) plus long context. Takes the HF model id as its second argument, so the same grid runs against any checkpoint. |
| `lc_rerun.sh` | The 100K/150K levels against an already-running container, with `GLLM_TIMEOUT=1800`. |
| `recheck_run.sh` | 300 s steady-state re-measure of the 8K/10K cells at concurrency 32-64, where a 60 s window under-reports. |
| `scaling_table.py` | Builds `benchmarks/SCALING-GRID.md` from every guidellm run: aggregate tok/s, per-stream tok/s, latency, and — for long context — total (prompt+output) tok/s and completed req/min. |
| `benchmarks/SCALING-GRID.md` | The generated grid. |
| `benchmarks/SCALING.log` | One line per scaling run (boot time, KV pool, end status, error count). |
| `20260822-qwen38-27b-concurrency-ceiling.html` | The write-up: where the card runs out of air, as charts. Self-contained apart from Google Fonts. |

`gllm.sh` gained the `pp512`/`pp2k`/`pp4k`/`pp8k`/`pp10k`/`lc100k`/`lc150k`
scenarios and a `GLLM_TIMEOUT` env var. Its default is still 60 s to keep the
older runs reproducible — raise it for anything above ~10K prompts.
| `mixed_load.py` | Concurrent text+image load generator. guidellm is text-only and cannot build a mixed-modality batch, which is the case that actually decides a multimodal config. |
| `benchmarks/PROD.log`, `benchmarks/serverlog-*.txt` | Per-config production run record and the engine log for each. |
| `20260818-qwen38-27b-speculation-crossover.html` | The full write-up with interactive charts. Self-contained apart from Google Fonts; no build step. |
| `benchmarks/RESULTS.log` | Every run, including the OOMs and the hang. |

**Do not use `docker rm -f` on a loaded model.** All three scripts stop with
`docker stop -t 90`. SIGKILL mid-CUDA-op is what wedges this card badly enough
to need a cold power cycle (`GPU-INCIDENT-RUNBOOK.md`).

### Updating a checkpoint — and why you may want to pin one

`HF_HUB_OFFLINE=0` in every compose here means the container re-resolves `main`
to the current upstream SHA on **each start**. Blobs are content-addressed, so
byte-identical weights are reused and only changed files download — a metadata
fix on a 23 GB repo updates in seconds.

```bash
docker compose -f docker-compose.vllm-27b-nvfp4-unsloth-mtp-rtx.yml down
docker compose --env-file ../../.env -f docker-compose.vllm-27b-nvfp4-unsloth-mtp-rtx.yml up -d
```

Verify it actually moved, rather than assuming:

```bash
# which revision the cache now points at
docker run --rm -v qwen38_huggingface_cache:/c alpine \
  cat /c/hub/models--unsloth--Qwen3.8-27B-NVFP4/refs/main

# and what the RUNNING container actually loaded
docker exec qwen38-27b-nvfp4-unsloth-mtp sh -c \
  'head -c 120 $(find /root/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots -name tokenizer.json | head -1)'
```

Old snapshots stay in the cache volume (disk grows by one snapshot per update,
though shared blobs are not duplicated).

⚠️ **`main` floats.** These composes pin the vLLM image but **not** the model
revision, so a restart silently loads whatever upstream published since — the
same drift hazard the image-pinning notes warn about, one layer down. On
2026-08-15 that worked in our favour; it can equally swap a working checkpoint
for a broken one during an unattended reboot. To pin, add to the vLLM args:

```yaml
      - --revision
      - 16b6615af3548b88e2d8e382457bc705b00479cf
```

### ⚠️ Community-quant hazard: Unsloth's 2048-token tokenizer truncation

**Fixed upstream 2026-08-15 in revision `16b6615a` — verified fixed here.**
Worth reading anyway, because it is the archetype of how a community repack
breaks in a way that does not look like a model bug.

Revisions of `unsloth/Qwen3.8-27B-NVFP4` before `16b6615a` shipped
`tokenizer.json` with `"truncation": {"max_length": 2048, …}` where official
Qwen has `"truncation": null`. Two failure modes
([discussion #10](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/discussions/10)):

- **Images >~1.4 MP fail loudly** — `ValueError: Mismatch in 'image' token count
  between text and 'input_ids'. Got ids=[2047] and text=[2280]`. Reads like a
  vLLM/processor bug; it is a metadata bug.
- **Text >2048 tokens truncates SILENTLY** — no error, no warning. A server
  advertising `max_model_len: 262144` simply stops reading past 2048.

The silent one is the dangerous one, and it is invisible to the obvious tests:
short prompts and small images both pass. Our own first NVFP4 vision check used
a 110 KB JPEG and a 900×420 PNG — under the threshold, so it would not have
caught this either way.

Verified on revision `16b6615a` (2026-08-15):

| Probe | Prompt tokens | Result |
|---|---|---|
| 1024×1024 image (1.05 MP) | 1,049 | ✅ |
| 1600×1200 image (1.92 MP) — reporter's regression case | 1,925 | ✅ |
| 2912×1632 image (4.75 MP) | **4,666** | ✅ |
| Long text, needle at 92% depth | 3,573 | ✅ recalled |
| Long text, needle at 92% depth | **10,613** | ✅ recalled |

4,666 tokens from one image and 10,613 from text are both far past 2048 —
exactly what the old revision could not do.

**Regression test if you ever change checkpoint or revision:** one image
≥1600×1200, and one text prompt >2048 tokens with the answer past the 2048 mark.
Check `usage.prompt_tokens` in the response, not just that a reply came back.

Related, same repo: [#11](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/discussions/11)
reports `AttributeError: 'MergedColumnParallelLinear' object has no attribute
'data'` on vLLM 0.27.1 **and** 0.25.1 — independent confirmation that the
`qwen38`-tagged image is mandatory, not merely preferred.

### The NVFP4 checkpoint is not what the 3.6 NVFP4 files assumed

`unsloth/Qwen3.8-27B-NVFP4` is **compressed-tensors / mixed-precision**, not
NVIDIA modelopt. Two consequences:

- **Do not pass `--quantization modelopt`** (as the 3.6 NVFP4 composes do) —
  vLLM reads `quant_method` off the checkpoint and that flag mis-dispatches it.
- Only the FFNs (`mlp.gate|up|down_proj`) are 4-bit, at `group_size 16`.
  Attention, the DeltaNet gates, `lm_head` and **the last 8 layers' MLPs
  (56–63)** stay FP8; the vision tower and the MTP head stay unquantized. It is
  23.44 GB vs 27.57 GB for official FP8 — only ~15% smaller. Judge it on speed;
  the VRAM saving is modest.

⚠️ **KV-cache trap:** the checkpoint *declares* an fp8 `kv_cache_scheme`. On
3.6's hybrid DeltaNet, fp8 KV under vLLM caused silent output corruption
(vllm-project/vllm#26646) — wrong answers, no crash. The scheme is embedded in
the checkpoint rather than passed on the command line, so vLLM may apply it
unbidden. The compose asserts `--kv-cache-dtype auto`; **verify it took** with
`docker logs qwen38-27b-nvfp4-unsloth 2>&1 | grep -i kv.cache.dtype` before
trusting any output.

---

## Official recipes — follow them, and note where they bite

- vLLM: <https://recipes.vllm.ai/Qwen/Qwen3.8-27B>
- SGLang: <https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B>
- Model card: <https://huggingface.co/Qwen/Qwen3.8-27B>

Three places where following the recipe overrides what this repo learned on 3.6:

1. **MTP method is `mtp`, not `qwen3_next_mtp`**, at `num_speculative_tokens: 3`.
   Our image registers `mtp`, `qwen3_5_mtp` *and* `qwen3_next_mtp` as distinct
   methods — `qwen3_next_mtp` belongs to Qwen3-**Next**, a different model, so
   guessing wrong is a silent mis-dispatch rather than a clear error. MTP also
   needs **vLLM 0.27.2+** for the gated-delta-net speculative fix; plain serving
   is fine on 0.27.1.
2. **`--kv-cache-dtype fp8` is the recipe default** — see the verification above.
3. **NVFP4: the recipe uses `Inferact/Qwen3.8-27B-NVFP4`**, not Unsloth's.

There is **no `nvidia/Qwen3.8-*` checkpoint** (the API 404/401s, while
`nvidia/Qwen3.6-27B-NVFP4` and `nvidia/Qwen3.6-35B-A3B-NVFP4` both resolve).
NVIDIA shipped 3.6 NVFP4 some weeks after that base release, so one may still
appear; re-check with
`curl -s "https://huggingface.co/api/models?author=nvidia&search=Qwen3.8"`.
If it lands, do **not** assume it drops into the Unsloth compose — theirs is
compressed-tensors/mixed-precision needing no `--quantization`, whereas NVIDIA's
3.6 checkpoints were modelopt and needed `--quantization modelopt`. Read the new
`config.json`'s `quant_method` first.

### Sampling parameters (from the model card)

| Mode | Settings |
|---|---|
| Thinking | `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0` |
| Instruct / non-thinking | `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5` |

`generation_config.json` ships `temperature=1.0, top_p=0.95, top_k=20`, which
vLLM applies — so **thinking-mode clients need send nothing**. Non-thinking mode
differs on three values the server cannot infer; the client must send them.
`presence_penalty` may be raised toward 2 to curb endless repetition, at some
risk of language mixing.

For agentic work the card recommends generous output budgets within a 1M
context: 262,144 tokens for reasoning, 131,072 for the final response. It also
warns that **lower reasoning effort does not always reduce end-to-end agentic
latency** — thinner analysis causes retries.

### 1M context via YaRN

262,144 is native; 1M needs RoPE scaling. Not configured in any compose here:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --max-model-len 1000000 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true,
    "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000,
    "partial_rotary_factor": 0.25, "factor": 4.0,
    "original_max_position_embeddings": 262144}}}'
```

YaRN is a static factor — it scales all positions, so it costs short-context
accuracy. Enable it only when you actually need >262K.

---

## Still open

**Concurrency-test coverage.** 8 configs were run through the full production
suite (guidellm chat sweep to concurrency 64 + mixed text+image to 32):
DSpark-7+prob, DSpark-7+prob+mnbt16k, DSpark-7+adaptive, DSpark-3, MTP-1, MTP-2,
MTP-2+prob, and no-speculation. **Not** concurrency-tested, because each is a
near-duplicate of one that was and single-stream rank proved to be a poor
predictor anyway: DSpark-5/6 variants (rows 2, 4, 7), DSpark-4 (row 8),
DSpark-7 plain (row 10), `dspark_draft_topk` (row 6), Unsloth NVFP4 (rows 11,
15) and FP8 (row 16). The two checkpoint rows are the real gap — Unsloth W4A16
and first-party FP8 might batch differently from W4A4, and nothing here rules
that out. Still true after the 2026-08-22 scaling grid, which is
4-bit only.

Closed on 2026-08-18: correctness under speculation (tools, vision, 43K needle,
temp-0 determinism — all ✅); the W4A4-vs-W4A16 question; MTP depth on a second
checkpoint (it does not transfer between checkpoints); and — the expensive
lesson — whether single-stream rank predicts production rank (it does not; the
production winner ranks 5th single-stream and the single-stream winner ranks
last under load). What remains:

1. **MTP depth under concurrency is not fully swept.** Depth 1 and 2 were
   measured; depth 2 + `probabilistic` won at concurrency 8-30 and depth 1 won
   at 55. Depth 3+ was never tried *at concurrency*, and the trend (shorter
   draft wins as load rises) says it will lose — but that is inference, not
   measurement. `probabilistic` was also only combined with depth 2.

2. **`--async-scheduling` + `probabilistic` hang.** Not isolated to one flag,
   not reported upstream, and worth ~+10% if it can be fixed — the two flags are
   each worth about the same and currently cannot be combined. Re-test on the
   next image repin.

3. **Long context beyond ~52K.** Needle *recall* is exact at 43,247 tokens
   under speculation, but the 262K ceiling is still unprobed for correctness, as
   is YaRN-extended 1M. Note the production config's KV pool is 967,870 tokens
   (3.69x full context) — DSpark's draft slots cost about half the pool vs no
   speculation. Long-context *throughput* is no longer open: the 2026-08-22
   scaling grid measures 100K and 150K at concurrency 1-16 and finds the card
   prefill-saturated at ~6,000 / ~5,000 total tok/s, with concurrency buying
   latency rather than throughput (~4 req/min at 100K, ~2 at 150K).

4. **SGLang.** Untested here, and it is the one stack with numbers that might
   beat ours: `gittensor` reports 147.9 tok/s on a **5090** with SGLang + their
   DSpark-NVFP4 drafter — the same drafter that will not load in vLLM. Their
   number is greedy/thinking-off, so compare it against our 112.2 greedy, not
   our 111.8 sampled. `lmsysorg/sglang:qwen38-27b` is pulled and ready.

5. **The Inferact W4A4 checkpoint.** `Inferact/Qwen3.8-27B-NVFP4` is the
   official recipe's choice and is also W4A4 modelopt, but with a *different*
   exclusion list than gittensor's (it excludes `in_proj_qkv`/`in_proj_z` too,
   and ships a separate `nvfp4_experts_mtp.safetensors`). Untested — it could be
   marginally faster or slower than 77.1.

6. **MTP depth for code workloads.** Acceptance is strongly workload-dependent
   (mean 2.2 on prose vs **4.86** on structured output on the *same* server), so
   the block-size and depth optima measured on the prose prompt should not be
   trusted for an agentic/coding endpoint. Re-sweep against a real code trace.

7. **`reasoning_effort` non-monotonicity** — `medium` beat `low` on TTFT.
   Needs more prompts to tell signal from noise.

8. **`thinking_token_budget`** — the exact logits-level cap from 3.6, not yet
   re-measured here.

9. **DFlash.** Still no `z-lab/Qwen3.8-27B-DFlash`. vLLM registers `dflash` as a
   method and the DSpark drafters carry a `dflash_config` block, so the path
   exists if a checkpoint appears.

10. **llama.cpp** compose is an untested scaffold.
