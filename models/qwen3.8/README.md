# Qwen3.8 Family

**STATUS: released 2026-08-14. NVFP4 + MTP is the production config —
97.9 tok/s, tools, long-context recall and vision all verified.**
`Qwen/Qwen3.8-27B` and `Qwen/Qwen3.8-27B-FP8` are both first-party and public
(Apache 2.0). All composes track the
[official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) except where
noted, and differ from each other only by checkpoint (+ `--speculative-config`
on the MTP variants), so every pairwise comparison is single-variable.

**Currently serving on the RTX PRO 6000 at `localhost:11484`:**
`docker-compose.vllm-27b-nvfp4-unsloth-mtp-rtx.yml` — 97.9 tok/s.
Verified: `vllm-27b-fp8-rtx`, `vllm-27b-nvfp4-unsloth-rtx`,
`vllm-27b-nvfp4-unsloth-mtp-rtx`. The rest are **UNVERIFIED** scaffolds and say
so in their headers.

Two gotchas that will bite you first — details below:
`enable_thinking: false` **degrades multi-step arithmetic** (use
`reasoning_effort: low` instead), and `test_chat.py --no-think` is a **no-op**
on this family.

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
| SGLang | `lmsysorg/sglang:qwen38` | untested here; tag was cut when only the 2.4T flagship was public |
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

### Throughput — and MTP, which is the whole ballgame

`test_chat.py --runs 3 --warmup`, all on the same card. TTFT is dominated by
thinking, not prefill, so it is not comparable to a non-reasoning model's TTFT.

| Config | tok/s | vs FP8 |
|---|---|---|
| FP8 (recipe) | 45.6 | — |
| NVFP4 Unsloth | 58.7 | +29% |
| **NVFP4 + MTP depth 2** | **97.9** | **+115%** |

**MTP depth sweep** on the NVFP4 checkpoint (one full load + benchmark per
depth). `num_speculative_tokens`:

| Depth | tok/s | vs no-MTP | Mean accept length | Per-position acceptance | Avg draft acceptance |
|---|---|---|---|---|---|
| off | 58.7 | — | — | — | — |
| 1 | 85.5 | +46% | 1.74 | 0.739 | 73.9% |
| **2** | **97.9** | **+67%** | 2.19 | 0.710 / 0.479 | 59.4% |
| 3 | 96.1 | +64% | 2.37 | 0.661 / 0.438 / **0.268** | 45.6% |

**The recipe's depth 3 is past the peak — this directory ships depth 2.** The
per-position column is the mechanism: acceptance decays hard per draft slot, and
at depth 3 the third token lands only **26.8%** of the time, so it drafts
130.8 tok/s to accept 59.6 and throws away two thirds of the draft compute.
Depth 2 drafts 93.2 to accept 55.4 and wins. Note acceptance *rate* falls
monotonically (73.9 → 59.4 → 45.6%) while mean accepted *length* rises
(1.74 → 2.19 → 2.37); throughput tracks the product until slot-3 decay dominates.

vLLM warns at startup that `num_speculative_tokens > 1` re-runs the single MTP
layer and "may result in lower acceptance rate". Correct about the mechanism,
wrong as a verdict — depth 2 still beats depth 1 by 14%.

**MTP costs no extra weight VRAM** (`model_mtp.safetensors` is referenced from
the safetensors index, so vLLM loads it even with speculation off) but it does
cost KV pool: 1,899,790 → 1,619,037 tokens at depth 2, about −15%.

**A prior that did NOT transfer:** Unsloth leaves the MTP head in BF16
(`re:^mtp.*` in the quant ignore list) against a 4-bit NVFP4 target — the widest
drafter/target precision mismatch here. On 3.6, that kind of divergence broke
speculation (DFlash stable on FP8, unstable on INT4). It does not reproduce:
73.9% acceptance at depth 1. Do not assume the 3.6 finding applies to 3.8.

⚠️ **Prose prompt only.** On 3.6 the optimal draft length flipped ~2x between
prose and code, and `test_chat.py`'s default prompt is close to spec-decode's
worst case. Re-sweep for coding workloads before trusting depth 2 there.

Thinking-mode sweep on the same prompt (3 runs each, *lightly loaded* — another
agent was sharing the card, so treat these as relative not absolute):

| Mode | TTFT | tokens | tok/s | reasoning chars |
|---|---|---|---|---|
| thinking on (default `xhigh`) | 16584 ms | 2048 (hit cap) | 45.3 | 3588 |
| `reasoning_effort: low` | 6511 ms | 1525 | 44.9 | 1331 |
| `reasoning_effort: medium` | 3472 ms | 1430 | 44.5 | 723 |
| `enable_thinking: false` | **79 ms** | 1329 | 44.1 | 0 |

**Decode throughput is flat at ~44–45 tok/s across every mode.** Thinking costs
TTFT and token count, never tok/s. The default `xhigh` burns ~3.6K characters of
reasoning and 16.5 s before the first visible token — set `reasoning_effort`
explicitly on any latency-sensitive endpoint.

**`reasoning_effort` is not monotonic.** `medium` came out *faster* than `low`
(3.5 s vs 6.5 s, and roughly half the reasoning). Both are prompt-level nudges,
not hard limits, so this may be prompt-specific — needs a larger sample before
anyone relies on `low` being the cheapest setting.

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

| File | Checkpoint | Status |
|---|---|---|
| `docker-compose.vllm-27b-fp8-rtx.yml` | `Qwen/Qwen3.8-27B-FP8` (first-party) | ✅ **VERIFIED** — tracks the official recipe. 45.6 tok/s, tools 4/4, vision working. Use this. |
| `docker-compose.vllm-27b-nvfp4-unsloth-mtp-rtx.yml` | `unsloth/Qwen3.8-27B-NVFP4` (**community**) | ✅ **VERIFIED — FASTEST. 97.9 tok/s** at MTP depth 2. The current default. |
| `docker-compose.vllm-27b-nvfp4-unsloth-rtx.yml` | `unsloth/Qwen3.8-27B-NVFP4` (**community**) | ✅ **VERIFIED.** 58.7 tok/s, tools 4/4, vision OK, needle-exact to ~51.7K. |
| `docker-compose.vllm-27b-bf16-rtx.yml` | `Qwen/Qwen3.8-27B` (first-party) | Unverified. ~54 GB. Only worth it as a quality reference vs FP8. |
| `docker-compose.vllm-27b-fp8-mtp-rtx.yml` | `Qwen/Qwen3.8-27B-FP8` (first-party) | Unverified. A/B partner for the FP8 baseline; sweep the depth. |
| `docker-compose.sglang-27b-rtx.yml` | `Qwen/Qwen3.8-27B` (first-party) | Unverified. Runtime A/B, NEXTN speculation. |
| `docker-compose.llama-27b-q4-rtx.yml` | `unsloth/Qwen3.8-27B-GGUF` (**community**) | Unverified. GGUF conversion confirmed to exist (`Qwen3.8-27B-UD-Q4_K_XL.gguf`). |

Every unverified file carries a header block listing exactly which assumptions
it makes.

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

Done on release day: FP8 booted, benchmarked, tool-tested; config and chat
template verified; NVFP4/GGUF checkpoints identified and their quant schemes
read. What remains:

1. **Long context beyond ~52K.** Needle retrieval is verified exact to ~51.7K
   tokens, but the 262K ceiling and the 1.67M-token KV pool are not probed near
   their limits. Also untested: YaRN-extended 1M context.

2. **Correctness under MTP.** Throughput is swept; correctness is not. Tools,
   vision and needle recall have **not** been re-run with speculation enabled,
   and on 3.6 MTP crashed the engine on mixed-modality batches — with vision
   enabled here, that path is reachable. Highest-value gap.

3. **MTP on the FP8 checkpoint.** Never booted, and it is the better-conditioned
   experiment: Qwen quantizes its MTP head too (22 tensors incl.
   `weight_scale_inv`), so drafter and target precision match, unlike Unsloth's
   BF16 head against a 4-bit target.

4. **MTP depth for code workloads** — see the prose-only caveat above.

4. **`reasoning_effort` non-monotonicity** — `medium` beat `low` on TTFT and
   reasoning length. Needs more prompts and runs to tell signal from noise.

5. **`thinking_token_budget`** — the exact logits-level cap from 3.6, not yet
   re-measured here. Re-run that sweep.

6. **DFlash.** No `z-lab/Qwen3.8-27B-DFlash` drafter exists yet. Note from 3.6
   that DFlash stability was quant-dependent (FP8 fine, INT4 not), so re-test
   rather than assuming.

7. **SGLang and llama.cpp** composes are untested scaffolds.
