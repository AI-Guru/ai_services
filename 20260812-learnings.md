# 2026-08-12 — Qwen3.6 serving: what we did and what we learned

One day on the RTX PRO 6000 (96 GB, SM120), all on `nvidia/Qwen3.6-*-NVFP4`.
Started as "drop a pip hack from one compose file", ended as a production
serving review that overturned two of its own recommendations along the way.

Commits: `59fe027`, `ad8b118`, `adb7305`, `381751e`, `810c754`, `9cb4a4f`.

---

## Headline numbers

| Config | before | after |
|---|---:|---:|
| 35B NVFP4 + MTP (test_scenarios overall) | 367.7 | **393.8** tok/s |
| 27B NVFP4 text-only + MTP (test_chat) | 95.9 | **138.1** tok/s |
| 27B NVFP4 vision, live endpoint | 138.3 (crashed) | **71.1** (stable) |
| 27B NVFP4 aggregate, batching not speculation | ~73 single | **586-608** tok/s @c16-32 |

Cold start across the board: 5-7 min → 150-230 s.

---

## 1. The pip-upgrade entrypoint was the root of it

Every NVFP4 compose ran `vllm/vllm-openai:nightly` behind an entrypoint that
`pip install -U`'d vLLM nightly + flashinfer on **every container start**. That
is a container which re-resolves its own dependencies on each boot, so a reboot
months later silently loads different software than what was benchmarked. It is
exactly how a colleague's deploy broke.

`v0.27.1` ships everything the hack was chasing — vllm 0.27.1, flashinfer
python/cubin/jit-cache 0.6.16.post3 (cu130), torch 2.13.0. Pinning it and
deleting the entrypoint gave **+4.5% with identical flags** and cut boot to
150 s.

**Learned:** `flashinfer-cubin 0.6.16.post3` exists only *inside* the image — it
is not on PyPI or flashinfer.ai (both stop at 0.6.13). So an entrypoint that
pins cubin to the installed version can never resolve. Reproduced live on our
own machine: one un-migrated config produced a three-way skew (python 0.6.17 /
jit-cache 0.6.18.dev / cubin 0.6.13).

**Debunked:** "the entrypoint ignores the envs." Compose `environment:` vars
reach an entrypoint shell normally (tested). `$${VAR}` = container expands,
`${VAR}` = compose expands at parse time. Both work; they just expand in
different places.

---

## 2. MTP is fast, and unusable in production

The official recipe's `mtp` method beat our `qwen3_next_mtp` N=1 by 27%, and
N=4 beat the recipe's N=3 by a further 8% on the dense 27B. The old comment
claiming "one MTP layer, so N>1 is unsupported" was simply wrong — vLLM applies
the head recursively.

Then load testing found **two independent engine-killing bugs**, both needing
only **concurrency 2**:

| Trigger | Affects | Evidence |
|---|---|---|
| Mixed modality batch (text + image together) | `-mtp-vision-` | 24-27/48 HTTP 500; 15/24 at conc 2 |
| Batched long sequences (~26K in) | `-mtp-textonly-` **and** vision | 12/12 at conc 4; 4/4 at conc 2 |

Controls make it airtight: 384 single-modality requests at up to 32-way
concurrency pass; one long request passes; 96 short prompts at 32-way pass; and
the identical long load passes 12/12 with speculation removed. Failure is
`CUDA error: an illegal memory access` → `EngineDeadError`, container exits,
`restart: always` reloads — ~4 minutes of downtime per occurrence. The GPU
survives (no Xid).

This is [vllm#40756](https://github.com/vllm-project/vllm/issues/40756)
reproducing on v0.27.1 + NVFP4 (filed against v0.19.1 + FP8), and a multimodal
trigger that appears **unreported upstream**.

**Learned the hard way:** single-stream benchmarks hid both. `test_chat` and
`test_scenarios` top out ~3-16K tokens at concurrency 1. We recommended and
committed a config that dies at two concurrent users, on the strength of a
138 tok/s number. **Any change touching speculative decoding needs a mixed-load
hammer before it goes live.**

---

## 3. Batching beats speculation by ~4×

| lever | gain |
|---|---|
| MTP speculative decoding | ~1.9× single stream |
| `--max-num-seqs 4 → 256` | **7-8×** aggregate (73 → 586 tok/s) |

Speculation optimises latency for one user; batching optimises tokens per GPU
per second, which is what a serving business sells. Our configs were all tuned
at `--max-num-seqs 4` — a single-user setting on a multi-tenant card.

---

## 4. Runtime comparison

| Runtime | single-stream | stability | verdict |
|---|---:|---|---|
| vLLM 0.27.1 no-MTP, `max-num-seqs 256` | 71.1 | passed everything | **fleet default** |
| vLLM 0.27.1 MTP N=4 | 138.1 | dies at conc 2, two ways | never, multi-tenant |
| SGLang 0.5.17 NEXTN (tuned) | 123.1 | passed everything | optional latency tier |
| TensorRT-LLM 1.3.0rc8 | — | cannot load the model | not an option |

SGLang runs the speculative decoding vLLM cannot — NEXTN survived every load
that kills vLLM MTP, while serving vision. But its advantage is narrow:

- short prompts, ≤16 concurrent → SGLang (+46% codegen @c8, +25% @c16)
- long prompts (8K+) → vLLM at every concurrency, SGLang saturates earlier
- ≥32 concurrent → vLLM

TensorRT-LLM ships transformers 4.57.3; these checkpoints declare
`model_type: qwen3_5` (needs ≥5.2), so it fails before quantization is even
considered. Upstream support is an open request
([#12321](https://github.com/NVIDIA/TensorRT-LLM/issues/12321)).

---

## 5. Traps worth remembering

**Flags that look right and aren't**

- `--kv-cache-dtype fp8` on the **35B** costs **19%** (318.0 vs 393.8) despite
  the official recipe listing it — plus an older silent-corruption report
  (vllm#26646). On the **27B** it is free (the checkpoint auto-selects fp8 KV),
  and on **SGLang** it is a 25% *win* (doubles the KV cache). Same flag, three
  different answers. Never port it blind.
- `--quantization modelopt` should just be dropped; vLLM auto-detects
  `modelopt_mixed`. The explicit 0.27.x spelling would be `modelopt_fp4`.
- MTP depth does **not** transfer between models: N=4 for the dense 27B, N=3 for
  the 35B MoE, where N=4-5 is ~10% *slower*. Cheap 3B-active decode means draft
  overhead outgrows the win sooner.
- The kv-cache flag changes the **shape** of the N curve, not just its height
  (with fp8 KV the 27B N=3 looked like a sharp peak; without it N=3 and N=4
  tie). Sweep N only after the rest of the config is settled.
- SGLang's cookbook `--attention-backend trtllm_mha` is **SM100-only** and
  refuses to start on SM120. Use `triton`.

**Things that lie to you**

- **`/health` returned 200 through every crash** while all inference 500'd. A
  broken replica stays in the load-balancer pool. Probe with a real completion.
- **5xx are logged at `INFO`** (uvicorn access lines) — `grep ERROR` never finds
  them. Meanwhile startup emits benign `[ERROR]` lines from transformers, so
  alerting on log level pages on every boot. Alert on `" 5[0-9]{2} "` and
  `RestartCount`.
- vLLM 0.27.1 logs `no registered multimodal processor; running in text-only
  mode` **while ingesting images normally**. Verify vision with a prompt-token
  diff (508 with image vs 22 without), not by keyword-matching a description —
  a text-only model can guess "Terminator" from a plausible prompt.
- `test_vision.sh` passing is **not** proof of vision; it keyword-matches.
- SGLang needs `cap_add: SYS_PTRACE` in Docker. Its CUDA IPC transport uses
  `pidfd_getfd(2)`, denied by the default seccomp profile. Without it the server
  boots, loads weights, captures CUDA graphs, reports healthy — then 500s every
  request. Looks like a broken runtime; is a container privilege.
- `trtllm-serve` 1.3.0rc8 needs the `serve` **subcommand**. The nine
  `models/nemotron/docker-compose.trtllm-*.yml` files omit it and would not
  start on that image tag.

**Method**

- A `docker compose down` removes the shared network; stopped containers then
  hold a stale network ID and fail to start with
  `network <id> not found`. Fix: `docker rm -f` the container and recreate. Bit
  us three times.
- A benchmark harness that boots a container without stopping whatever already
  holds the GPU produces a silent timeout, not an error. Cost one void run.
- Correctness probes must discriminate. Our first fp8-KV check asked for a
  capital city and got a thinking preamble — it verified nothing. Needle recall
  at increasing depth is the test that would actually catch KV-cache damage.

---

## 6. Config layout now

|  | text-only | vision |
|---|---|---|
| **MTP** | `...nvidia-mtp-textonly-rtx.yml` — 138.1, short prompts only | `...nvidia-mtp-vision-rtx.yml` — ⚠️ single-modality only |
| **no MTP** | `...nvidia-rtx.yml` — 71.1 | `...nvidia-vision-rtx.yml` — 71.1, **safe live endpoint** |
| **throughput** | — | `...nvidia-vision-parallel-rtx.yml` — 586-608 @c16-32 |
| **SGLang** | — | `sglang-27b-nvfp4-nvidia-rtx.yml` — pinned v0.5.17 |
| **TRT-LLM** | — | `trtllm-27b-nvfp4-nvidia-rtx.yml` — record of why it fails |

Naming caveat: the NVFP4 group treats a plain name as text-only, the FP8 group
treats it as vision. The two families contradict each other.

---

## 7. Open items

1. **The 35B on 11438 has never been load-tested** and is live with `mtp` N=3.
   Everything above is the 27B. Different architecture, so the bugs may or may
   not transfer — but its 393.8 came from single-stream runs, the same blind
   spot that produced a config dying at concurrency 2. **Highest-value next
   step, ~30 min.**
2. **~10 compose files still carry the pip-hack entrypoint** — `laguna-s`,
   `soofi-s`, `ling-3.0-flash`, and the 27B/35B no-MTP configs.
3. **File the multimodal MTP bug upstream.** Deterministic, reproducible in
   seconds, clean controls, on a current release, and apparently unreported.
4. **SGLang's `max_running_requests` pins at 42** regardless of request (256 and
   1024 both give 42). Likely the c32 falloff. Unexplained.
5. **fp8 KV verified to 17.4K context, not the 32K ceiling** — re-verify before
   pushing long-context traffic through SGLang.
6. **Four `0.0` cells in the long-prompt table** need a longer `MAX_SECONDS`.
7. **The 35B serves no tool-calling parser** (27B uses `qwen3_coder`; the recipe
   suggests `qwen3_xml`). Untouched, needs a `test_tools.py` run.
8. **Production hardening, runtime-independent:** real inference health probe,
   alerting on 5xx + `RestartCount`, ≥2 replicas per model (4-min cold start =
   4-min outage), and pin every image tag.

---

## 8. What I'd tell someone starting tomorrow

Pin your images and delete every `pip install` from an entrypoint — a serving
container that mutates itself on boot is not a config, it is a lottery ticket.
Then load-test before you believe any number: today's two worst findings were
both invisible to single-stream benchmarks, and both configs had already been
recommended, committed, and in one case deployed on the strength of those
numbers. Speculative decoding is the most fragile thing in the stack and the
least valuable at scale; batching is boring, dull, and worth four times more.
