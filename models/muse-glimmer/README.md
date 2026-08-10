# Muse Glimmer-30B — Meta's open agentic model on one RTX PRO 6000

`meta-models/Muse-Glimmer-30B`: a dense 29.6B causal transformer with a ~1.8B ViT-G/14
perception encoder, distilled from Muse Spark and built for local agentic work.
52 layers, `[local, local, local, global]` attention with a 2048 sliding window,
GQA 32/2, 131072 context, text+image in, Apache 2.0. Released 2026-08-10.

Benchmarked on **three engines** — vLLM, SGLang and llama.cpp — from first-party Meta
checkpoints only (BF16 safetensors and Meta's own GGUF exports; no community requants).

All three needed day-0 builds, and none of them a released version: vLLM support is
[PR #51655](https://github.com/vllm-project/vllm/pull/51655) (**open**), SGLang's is
[PR #34262](https://github.com/sgl-project/sglang/pull/34262) (**open**), and
llama.cpp's is [PR #26841](https://github.com/ggml-org/llama.cpp/pull/26841), merged
2026-08-10 11:07 UTC — 16 minutes *after* release `b10338` was cut, so the newest
tagged llama.cpp release does not contain it either. vLLM and SGLang publish purpose-
built image tags; llama.cpp is built from master here (`Dockerfile.llama-muse-glimmer`).

**Fastest configuration measured: llama.cpp K-Quant-17GB + DFlash** — 110 tok/s on the
repo's standard prose benchmark (4.0x the vLLM BF16 baseline) and **250 tok/s on code
generation** (9.1x). The 2x spread between those two is the single most important
finding here; see "the DFlash speedup is dominated by what you ask it".

## Results — three engines, RTX PRO 6000 (96 GB), single stream

All five configurations measured with the same command and the same sampling
(Meta's recommended temperature 1.0 / top_p 0.95 / top_k 64), 3 runs + warmup:

```bash
python3 ../shared/test_chat.py --base-url http://localhost:<port>/v1 \
  --model <served-name> --runs 3 --warmup --no-think
```

| Engine | Weights | Spec | Port | **tok/s** | vs BF16 | TTFT→content |
|---|---|---|---|---|---|---|
| vLLM | BF16 55.49 GiB | — | 11441 | 27.4 | 1.00x | 30026 ms |
| SGLang | BF16 55.49 GiB | — | 11443 | 27.9 | 1.02x | 26278 ms |
| SGLang | BF16 55.49 GiB | DFlash b=5 | 11444 | 49.2 | 1.80x | 16633 ms |
| llama.cpp | K-Quant-Dynamic 19.65 GB | — | — | 68.7 | 2.51x | 11562 ms |
| llama.cpp | K-Quant-Dynamic 19.65 GB | DFlash n=15 | — | 98.0 | 3.58x | 7896 ms |
| llama.cpp | **K-Quant-17GB** 16.76 GB | — | 11445 | 80.3 | 2.93x | — |
| **llama.cpp** | **K-Quant-17GB** 16.76 GB | **DFlash n=15** | 11446 | **110.0** | **4.01x** | — |

vLLM + DFlash is missing because it does not work — see the DFlash section.

**These are all on `test_chat.py`'s default prompt, which is the worst case for
speculation.** The same 11446 config reaches **250.1 tok/s on code generation**
(9.1x). See "The DFlash speedup is dominated by what you ask it" below — that
section matters more than this table for choosing a config.

**The whole table is one ratio: bytes moved per token.** A dense 30B is memory-bound
on this card, so throughput tracks (bytes per weight) x (weight-passes per token), and
almost nothing else:

- **Engine choice is nearly irrelevant at BF16.** vLLM 27.4 vs SGLang 27.9 — a 2% gap
  between two completely different runtimes, both streaming the same 55.49 GiB per
  token. ~1.8 TB/s / 55.49 GiB ≈ 30 tok/s, so both sit at ~91-93% of the roofline.
  There is nothing to tune here.
- **Quantization attacks bytes-per-pass**, and the gain tracks the byte ratio almost
  exactly: 55.49 GiB → 19.65 GB (2.9x) delivers 2.51x; 19.65 GB → 16.76 GB (1.17x)
  delivers 1.17x (68.7 → 80.3 tok/s). This is the most predictable lever available.
- **DFlash attacks passes-per-token**, and its gain is *not* predictable — it depends
  entirely on how guessable the output is (1.37x on prose, 3.1x on code).
- **The two levers compose**: 2.93x (quant) x 1.37x (DFlash) = 4.01x, and
  27.4 x 4.01 = 110.0. Exactly the measured value.

**TTFT collapses as a side effect.** 30.0 s → 7.9 s is not a prefill improvement —
true prefill is ~183 ms throughout. It is the same reasoning tokens emitted 3.5x
faster. See the reasoning-strength section for why TTFT is dominated by reasoning.

### Per-engine detail

| | vLLM | SGLang | llama.cpp |
|---|---|---|---|
| Image | `vllm/vllm-openai:muse-glimmer-x86_64-cu130` | `lmsysorg/sglang:dev-muse-glimmer` | built here, `d2f8305` |
| Version | `0.1.dev19075+gd89ec6d6a` | `dev-muse-glimmer` | master @ 2026-08-10 12:13 UTC |
| Weight load | 16.3 s | 14.6 s | fast (mmap, 18.3 GiB) |
| KV cache | 29.97 GiB → 1,699,807 tok | 551,211 tok | `n_ctx` 131072 |
| Reasoning field | **`reasoning`** | `reasoning` | **`reasoning_content`** |
| Parser flags | `muse_glimmer` | `muse` | `--jinja` (built in) |
| DFlash | **broken** | works as published | works as published |
| Vision | verified | available (`--mm-feature-transport cpu`) | `mmproj-kquant.gguf` |
| Tool calls | verified | available | verified |

Note the **reasoning field name differs between engines** — vLLM and SGLang populate
`reasoning`, llama.cpp populates `reasoning_content`. Anything consuming these
endpoints has to handle both. `test_chat.py` checks both names, so its numbers are
comparable across all three.

### DFlash acceptance, measured

| Engine | Draft len | Workload | Mean accepted len | Speedup |
|---|---|---|---|---|
| SGLang | 5 | prose | 1.65 – 2.35 | 1.76x |
| llama.cpp | 15 | prose | 2.28 – 2.40 | 1.37x |
| llama.cpp | 3 | code | 3.27 | 2.10x |
| llama.cpp | 15 | code | **5.55** | **3.11x** |

Mean accepted length is the figure to compare (the raw "accept rate" the two engines
print is defined differently — SGLang per verify step, llama.cpp per drafted token —
so those numbers are not comparable across engines).

The last two rows are the point: **the drafter is 2.7x better at guessing code than
prose**, and that single fact drives everything about how to configure DFlash.

### The DFlash speedup is dominated by WHAT YOU ASK IT, not by sampling

The single most important result in this file. Same server, same build, same sampling
— only the prompt changes (K-Quant-17GB, `n_max` 8, temp 1.0):

| Prompt category | tok/s |
|---|---|
| Code generation | **215.0** |
| RAG / summarization | 199.1 |
| Open-ended reasoning (`test_chat.py` default prompt) | 110.0 |

**A 2x spread from the prompt alone.** Speculation pays in proportion to how
predictable the output is; code and quoted-source summarization are highly
predictable, open-ended prose is not. The llama.cpp DFlash PR's own SpeedBench shows
the same shape across categories (rag 4.07x / coding 3.11x / qa 2.20x).

Two consequences, both of which cost us a wrong answer before we measured:

**1. `test_chat.py`'s default prompt is the WORST CASE for this model.** Every
DFlash row in the results table understates what an agentic workload will see. The
repo-convention benchmark is a fine cross-engine control precisely because it is
stable, but it is not representative of what Muse Glimmer is for.

**2. The optimal draft length flips with workload.** Measured on K-Quant-17GB @ temp 1.0:

| Workload | n_max=3 | n_max=8 | n_max=15 | mean accepted len |
|---|---|---|---|---|
| Reasoning chat | **110.0** | — | 98.0 | 2.04 (at n_max 3) |
| Code generation | 168.3 | 215.0 | **250.1** | 5.55 (at n_max 15) |

On code the drafter lands 5.55 tokens per verify step and long drafts win by **49%**;
on prose it lands 2.04 and the extra draft work is wasted, so short drafts win by 12%.
Tuning `n_max` on the reasoning prompt alone gives exactly the wrong answer for an
agentic coding model. **The composes default to `n_max` 15.**

**Peak measured: 250.1 tok/s** (codegen, K-Quant-17GB, DFlash `n_max` 15) — 9.1x the
vLLM BF16 baseline, and above Meta's published 233.4 tok/s.

### Sampling barely matters — tested and refuted

It is tempting to attribute the gap to Meta benchmarking at greedy while we sample.
Measured on K-Quant-17GB, greedy (`--temp 0 --top-k 1`) vs Meta's recommended
sampling:

| | baseline | DFlash n_max=3 | n_max=8 | n_max=15 |
|---|---|---|---|---|
| Greedy | 80.5 | 114.2 | 111.6 | 110.6 |
| temp 1.0 / top_p 0.95 / top_k 64 | 80.3 | 110.2 | — | — |

**Greedy buys ~3.6%**, and mean accepted length moves only 2.04 → 2.21. Sampling is
not the explanation; prompt category is. All headline numbers in this README therefore
use Meta's recommended sampling, which is how the model is actually served.

### Reasoning strength, and two benchmark caveats

This model has **no thinking on/off switch**. The chat template always emits
`Reasoning strength: <low|medium|high|xhigh>.` into the system message, defaulting to
`high`. Set it with `chat_template_kwargs: {"reasoning_strength": "low"}` — verified
working. It changes how many tokens get spent, not the decode rate:

| `reasoning_strength` | prefill TTFT | total | completion tokens | tok/s |
|---|---|---|---|---|
| `high` (default) | 185 ms | 58.3 s | 1598 | 27.4 |
| `medium` | 182 ms | 39.6 s | 1091 | 27.5 |
| `low` | 182 ms | 29.8 s | 822 | 27.6 |

1. **`test_chat.py --no-think` is a no-op here.** It sends a top-level
   `enable_thinking` field, which this model has no concept of. Every "no-think" run
   still reasoned for ~30 s. Same trap already recorded for models/ling-3.0-flash and
   models/soofi-s, different mechanism — there is no template flag to find, because
   reasoning is a *strength dial*, not a boolean.
2. **The 30026 ms "TTFT" is not prefill.** `test_chat.py` reports time-to-first-token
   including reasoning. True prefill is **~183 ms**, flat across all reasoning
   strengths. The tok/s figure is unaffected — `test_chat.py` takes
   `completion_tokens` from server-side `usage`, which counts reasoning tokens.

### Reasoning is exposed as `reasoning`, not `reasoning_content`

vLLM's `muse_glimmer` reasoning parser populates **`reasoning`** (the newer OpenAI
field), in both streaming deltas and non-streaming messages. `reasoning_content` is
always empty. `test_chat.py` happens to check both names, so it works either way, but
anything else reading this endpoint needs the right key.

Note also that a truncated response yields **neither** channel — the parser only emits
a channel once it closes, so `max_tokens` too small returns `content: null` with a
nonzero `completion_tokens`. At default `high` reasoning, budget a few thousand tokens.

## Getting it to load on a 30 GB-RAM host — two required flags

**This applies to vLLM AND SGLang, and is a property of the host, not the engine.**
SGLang fails with a byte-identical error and takes the same fix. llama.cpp is immune:
its largest single file is the 18.3 GiB GGUF, comfortably under the threshold.

The host has **30 GB RAM + 8 GB swap**; the model's first shard is a single **50 GB**
safetensors file. Both flags below are mandatory here and are baked into the composes.

**1. `--load-format=runai_streamer`.** The default loader dies with:

```
RuntimeError: unable to mmap 49950112952 bytes from file <...model-00001-of-00002.safetensors>:
Cannot allocate memory (12)
```

`safe_open` maps the shard `MAP_PRIVATE` **writable** (copy-on-write). Writable private
mappings are charged against `vm.overcommit_memory=0`'s heuristic, and a 50 GB
accountable mapping on a 38 GB box is refused outright. Measured, to rule out a real
shortage: a `PROT_READ` mmap of the same file **succeeds**; adding `PROT_WRITE` fails
with `Errno 12`. The pages are never dirtied — it is pure accounting.

Host-side alternatives, neither applied here (both need root, both change machine-wide
behaviour): `sysctl -w vm.overcommit_memory=1`, or adding ≥20 GB of swap.

**2. `--model-loader-extra-config={"concurrency":8,"memory_limit":6442450944}`.**
`runai_streamer` defaults to an **unbounded** host-RAM buffer, which just trades the
mmap error for the OOM killer. The engine reached ~28.8 GB anon-rss at ~18% of 1436
tensors and was SIGKILLed. vLLM reports this only as:

```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

with **no traceback and an empty failed-proc dict**. The container log contains nothing
useful — the evidence is in `journalctl -k`:

```
Out of memory: Killed process ... (VLLM::EngineCor) total-vm:295580428kB, anon-rss:28812352kB
```

`memory_limit` is in **bytes**. Treat "engine core failed with an empty dict and no
traceback" as "check the kernel log for an OOM kill" on any large model on this host.

Side benefit: `runai_streamer` loads 55.49 GiB in **16.3 s** (14.6 s on SGLang).

On SGLang, set the bound via the **environment variables**
`RUNAI_STREAMER_MEMORY_LIMIT` / `RUNAI_STREAMER_CONCURRENCY` — those are what the
streamer actually reads; vLLM's `--model-loader-extra-config` merely sets them.

### SGLang only: `pidfd_getfd: Operation not permitted`

SGLang's default `cuda_ipc` transport for multimodal features calls `pidfd_getfd`,
which Docker's default seccomp profile gates behind `CAP_SYS_PTRACE`. Without it the
**scheduler** dies on the first forward pass — long after the weights have loaded
successfully and the CUDA graphs have been captured — and the visible symptom is a
warmup `ReadTimeoutError` **600 seconds later**:

```
RuntimeError: pidfd_getfd: Operation not permitted        # the real cause, in the scheduler
urllib3.exceptions.ReadTimeoutError: ... (read timeout=600)   # what you actually see
```

Fixed with `--mm-feature-transport cpu`, which avoids the syscall entirely and needs
no extra container capability (vision still works, via a CPU copy). `cap_add:
[SYS_PTRACE]` is the alternative; the flag is less privilege for the same result.

### llama.cpp only: argument syntax

llama.cpp's parser does **not** accept `--flag=value`. Every flag and its value must
be separate argv entries or the server refuses to start:

```
error: invalid argument: --model=/models/muse-glimmer-30B-kquant-dynamic.gguf
```

### Three failure modes, three misleading symptoms

Worth internalising for any large model on this host — none of these say what they mean:

| Symptom | Actual cause | Where the evidence is |
|---|---|---|
| `unable to mmap ...: Cannot allocate memory (12)` | overcommit accounting on a writable-COW map | reproduce with `PROT_READ` vs `PROT_WRITE` |
| `Engine core initialization failed ... Failed core proc(s): {}` | OOM killer SIGKILLed the engine | `journalctl -k`, **not** the container log |
| warmup `ReadTimeoutError (read timeout=600)` | instant seccomp `EPERM` on `pidfd_getfd` | `Scheduler hit an exception` earlier in the log |

## Image, parsers, chat template

- **Image must be `vllm/vllm-openai:muse-glimmer-x86_64-cu130`**, not `:nightly`.
  The docker-hub `:nightly` here is vLLM 0.20.2 and does not know `muse_glimmer` at all.
  Same arch-specific-image pattern as the gemma4 images in the repo CLAUDE.md.
- **`--reasoning-parser muse_glimmer` and `--tool-call-parser muse_glimmer` must be used
  together.** The model emits neither `<think>` tags nor JSON tool calls. Every turn is
  a sequence of channel-scoped messages —
  `<|start|>assistant to=<recipient><|message|> … <|eom|>` — where `to=self` is
  reasoning and `to=<tool.fn>` is a tool call whose body is ATEM XML
  (`<atem:function_calls>`). Both parsers key off that framing, and the reasoning parser
  forces `skip_special_tokens=False` internally.
- **`tool_choice="auto"` only.** vLLM sets `supports_required_and_named=False` for this
  parser, so `"required"` and named tool choice get no JSON guided decoding.
- **No `--chat-template`.** The checkpoint's own template is correct. vLLM ships
  `examples/tool_chat_template_muse_glimmer.jinja`, but it renders images as `<|image|>`
  whereas the checkpoint's template and `config.json` (`image_token_id: 200092`) use
  `<|patch|>`.
- Meta's recommended sampling: `temperature 1.0`, `top_p 0.95`, `top_k 64`.

## DFlash speculative decoding — works on SGLang and llama.cpp, broken on vLLM

Meta ships a separate block-diffusion drafter that reads the target's residual stream
at layers [1,13,25,37,49] and proposes a 16-token block per forward pass. Two exports:

- `meta-models/Muse-Glimmer-30B-assistant` — BF16, 4.8 GiB, for SGLang/vLLM
- `dflash-kquant.gguf` — Meta's own quantized GGUF drafter, 1.63 GB, for llama.cpp

**On SGLang and llama.cpp both serve as published — no conversion, no config patching,
first try.** Measured 1.76x and 1.40x respectively (see the acceptance table above).

### vLLM: broken upstream, not forced

`docker-compose.vllm-30b-bf16-dflash-rtx.yml` is kept only as a record of how far it
gets. That the *same* checkpoint loads unmodified on SGLang makes this squarely a vLLM
integration defect. Four problems, the last fatal:

1. The drafter arch is registered unreachably. `registry.py` has
   `"MuseGlimmerAssistantModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM")`, but
   `EAGLEConfig(method="dflash")` rewrites every draft arch to `DFlash{arch}` unless it
   already starts with `DFlash`. So the registered name can never be reached:
   `ValidationError: Model architectures ['DFlashMuseGlimmerAssistantModel'] are not supported`.
   Worked around by rewriting the drafter config to `DFlashDraftModel`.
2. The drafter config ships **no `vocab_size`**, so it inherits Qwen3's 151936 against
   the target's 202048 and vLLM refuses the mismatch. Worked around.
3. `ValueError: DFlash sliding attention requires a window size configured in
   dflash_config.swa_window_size or the top-level sliding_window.`
4. Behind (3), the entire `dflash_config` schema is missing. vLLM maps this drafter onto
   its existing `qwen3_dflash` implementation, which reads `use_swa`, `swa_window_size`,
   `causal`, `mask_token_id`, `add_swa_attention_sink_bias`, `use_aux_hidden_state`.
   Meta's checkpoint uses a different schema (`layer_types`, `sliding_window`,
   `target_layer_ids`, `block_size`). **The PR never wired up a translation between
   them.**

Hand-authoring that dict was deliberately **not** done: `causal` and the sink-bias flag
change *output correctness*, not just startup, so a server that boots would prove
nothing about whether the drafts are right.

This matches the ecosystem. Meta/HF's own launch blog serves vLLM with
`--model-impl transformers` and says speculative decoding is documented for the
transformers and llama.cpp backends, **not** vLLM. vLLM
[issue #42068](https://github.com/vllm-project/vllm/issues/42068) separately reports
DFlash drafters being forced onto `TRITON_ATTN` by MTP backend propagation.

Revisit when PR #51655 merges with a real config translation. Until then, DFlash on
this model means SGLang or llama.cpp.

## Files

| File | Engine | Port | Status |
|---|---|---|---|
| `docker-compose.llama-30b-kquant-dflash-rtx.yml` | llama.cpp | 11446 | **fastest — 110 tok/s prose, 250 tok/s code** |
| `docker-compose.llama-30b-kquant-rtx.yml` | llama.cpp | 11445 | working, 80.3 tok/s |
| `docker-compose.sglang-30b-bf16-dflash-rtx.yml` | SGLang | 11444 | working, 49.2 tok/s |
| `docker-compose.sglang-30b-bf16-rtx.yml` | SGLang | 11443 | working, 27.9 tok/s |
| `docker-compose.vllm-30b-bf16-rtx.yml` | vLLM | 11441 | working, 27.4 tok/s |
| `docker-compose.vllm-30b-bf16-dflash-rtx.yml` | vLLM | 11442 | **broken upstream**, do not deploy |
| `Dockerfile.llama-muse-glimmer` | — | — | builds the llama.cpp image |

**Recommended default: the llama.cpp DFlash compose** — 4.0x the vLLM BF16 baseline on
prose and 9.1x on code, using Meta's own first-party GGUF exports and the drafter Meta
published.

Build the llama.cpp image before first use:

```bash
docker build -t llama.cpp-muse-glimmer:latest \
  -f Dockerfile.llama-muse-glimmer .
```

Storage — two docker volumes:
- `museglimmer_huggingface_cache` — BF16 target 59.5 GB + BF16 drafter 5.1 GB
- `museglimmer_gguf` — K-Quant-17GB 16.76 GB (served) + K-Quant-Dynamic 19.65 GB
  (benchmarked, removable) + mmproj 1.4 GB + DFlash drafter 1.63 GB

## Not measured

- **NVFP4.** `RadixArk/Muse-Glimmer-NVFP4` is third-party (unvetted per repo policy),
  but the SGLang cookbook marks it `verified: true` on this exact card and pairs it
  with DFlash + `--kv-cache-dtype fp8_e4m3`. LMSYS publishes **214 tok/s** batch-1 and
  403 tok/s aggregate at batch 8 for that combination on an RTX PRO 6000 — more than
  double our best measured result. This is the clear next experiment.
- **SGLang block-size tuning.** SGLang ran only at the cookbook's block 5. Given that
  llama.cpp's optimum swung 49% with draft length on code, sweeping
  `--speculative-dflash-block-size` on SGLang is likely worth a lot — untested.
- **SGLang on the GGUF or with a codegen prompt.** All SGLang numbers are BF16 on the
  prose prompt, i.e. its own worst case.
- **Concurrency scaling.** All numbers are single-stream. KV headroom is large
  (1.7M tokens on vLLM, 551k on SGLang).
- **Agentic/tool benchmarks.** `test_tools.py` / `test_scenarios.py` not run; tool
  calling verified functionally on vLLM and llama.cpp only.
- **llama.cpp vision.** `mmproj-kquant.gguf` is wired into both llama.cpp composes but
  image input was verified on vLLM only.
