# Ling-3.0-flash — 124B hybrid-attention MoE on one RTX PRO 6000

`inclusionAI/Ling-3.0-flash`: a ~124B-parameter MoE with a hybrid attention stack —
35 KDA (Kimi-Delta linear attention) layers interleaved with 7 MLA layers, 512 routed
experts + 1 shared, top-8 routing, and a multi-token-prediction head. vLLM support
landed in [PR #51045](https://github.com/vllm-project/vllm/pull/51045), merged to main
on 2026-08-05.

Served from the first-party INT4 checkpoint, `inclusionAI/Ling-3.0-flash-int4`
(77.0 GB), which keeps `model_type` and the full 262144 context and needs no config
overrides. The first-party `-fp8` build (128.4 GB) does not fit on a 96 GB card; the
first-party `-fp4` build (70.4 GB, MXFP4 experts) is untested here and declares
inclusionAI-specific config keys that stock vLLM may not parse.

## Results — INT4 + MTP, RTX PRO 6000 (96 GB), vLLM 0.26.1rc1.dev408

`docker-compose.vllm-124b-int4-mtp-rtx.yml`, TP1, `--max-model-len 262144`,
`--max-num-seqs 8`, MTP `num_speculative_tokens=1`.

| Metric | Value |
|---|---|
| Decode, thinking **off** | **173.8 tok/s** |
| TTFT, thinking **off** | **96 ms** |
| Decode, thinking **on** (`test_chat.py` avg of 3) | **148.6 tok/s** |
| TTFT to first *content* token, thinking on | 5119 ms avg (2044 ms warm) |
| MTP draft acceptance rate | **62.5–75.6%** |
| MTP mean acceptance length | **1.63–1.76** (ceiling 2.0 at k=1) |
| Weights resident | 71.87 GiB |
| KV cache | 9.67 GiB → 1,010,634 tokens (3.86x concurrency at 262144) |
| Model load | 727 s cold / ~50 s warm page cache |

Benchmark command (repo convention):

```bash
python3 ../shared/test_chat.py --base-url http://localhost:11481/v1 \
  --model ling-3.0-flash --runs 3 --warmup --no-think
```

Two caveats on those numbers, both measured rather than assumed:

- **`--no-think` is a no-op against this model.** `test_chat.py` sends `enable_thinking`
  as a top-level request field; vLLM only honours it inside `chat_template_kwargs`. Every
  "no-think" run still emitted a reasoning block (`think 8345 ms`). The thinking-off rows
  above were measured by sending `chat_template_kwargs: {"enable_thinking": false}`
  directly. The same trap is recorded for models/soofi-s.
- **The thinking-on TTFT is not prefill.** It is time-to-first-*content*-token, so it
  measures reasoning length. True prefill TTFT is 96 ms.

**Not measured: the no-MTP baseline.** The acceptance length of 1.63–1.76 means the
target model is producing ~1.7 tokens per forward pass instead of 1.0, so MTP is clearly
doing real work — but this was not run against a speculation-free control, and enabling
MTP also costs full CUDA graphs (below). Treat "MTP is worth it here" as strongly
indicated, not demonstrated.

## MTP notes

The MTP head is **layer 42 of the checkpoint itself** (`num_nextn_predict_layers: 1`),
not a separate drafter, so `--speculative-config` points `model` at the target. vLLM
logs `Detected MTP model. Sharing target model lm_head weights with the draft model.`

`num_speculative_tokens` is 1 because there is exactly one nextn layer.

vLLM accepts `"method": "bailing_hybrid_v3_mtp"` but deprecates it in favour of
`"mtp"`, rewriting it internally — the compose uses `mtp` directly.

Enabling MTP downgrades CUDA graphs:

```
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention
backend TritonMLABackend; setting cudagraph_mode=PIECEWISE
```

That is a real counterweight to the speculation gain and the main reason the no-MTP
baseline is worth running before concluding MTP wins.

## Operational notes

- **Port 11481.** 11441 is taken by `models/nemotron` (llama-nano-omni-30b maps
  `11441:11441`).
- **The docker-hub `vllm/vllm-openai:nightly` image ships vLLM 0.20.2**, which has
  BailingMoe V2/V2_5 but not V3. The entrypoint pip-upgrades to the wheels.vllm.ai
  nightly, as in `models/soofi-s/`. That upgrade is **unpinned** — two launches minutes
  apart picked up dev407 and dev408. Pin it if you need to reproduce a number exactly.
- The compose asserts `BailingMoeV3ForCausalLM` **and** `BailingMoeV3MTPModel` are in the
  model registry before `vllm serve`, so a stale wheel fails in seconds instead of after
  a 77 GB download and an 11-minute load.
- A benign transformers warning recurs on every start: `You are using a model of type
  bailing_hybrid to instantiate a model of type ''`. inclusionAI's bundled config class
  declares no `model_type`; vLLM reads the JSON value, which is the one that matters.
- Only the 7 MLA layers hold a per-token KV cache (~8 KB/token). The 35 KDA layers hold a
  fixed-size recurrent state **per sequence**, so `--max-num-seqs` drives memory more than
  context length does.
- Parsers: PR #51045 registers **both** the reasoning and tool-call parser as `ling3`.
