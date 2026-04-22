# Qwen3.6 Family

Two open-weight variants from Alibaba Qwen (Apache 2.0):

- **[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** — 35B-total / 3B-active MoE (fast, cheap to serve)
- **[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)** — 27B dense (stronger per-token, bandwidth-bound)

Both share the hybrid Gated DeltaNet + Gated Attention architecture and the same reasoning / tool-calling parsers (`qwen3` / `qwen3_coder`).

---

# Qwen3.6-35B-A3B

[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — 35B-total / 3B-active Mixture-of-Experts model from Alibaba Qwen.

## Architecture

- **Hybrid attention**: 30 Gated DeltaNet (linear attention) + 10 full Gated Attention layers in a repeating 3:1 pattern
- **MoE**: 256 experts, 8 routed + 1 shared active per token
- **Multimodal**: text, image, and video input (up to 224K video tokens)
- **Context**: 262K native, extensible to ~1M via YaRN
- **Multi-Token Prediction**: native MTP for speculative decoding
- **License**: Apache 2.0

## Quick start

```bash
# RTX PRO 6000 (96 GB) — primary config
docker compose -f docker-compose.vllm-35b-fp8-rtx.yml up -d

# Apple Silicon (24+ GB unified memory)
./run-qwen-mlx.sh --install
./run-qwen-mlx.sh --serve
```

API endpoint: `http://localhost:11435/v1` (vLLM/llama.cpp/SGLang), `http://localhost:11436/v1` (MLX), model name: `qwen3.6-35b`

## MLX (Apple Silicon)

Script: `run-qwen-mlx.sh` — uses `mlx-openai-server` with `qwen3.6_chat_template.jinja`, port **11436**.

| Model | Size | Memory |
|-------|-----:|-------:|
| `mlx-community/Qwen3.6-35B-A3B-4bit` *(default)* | ~20 GB | 24+ GB |
| `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ` | ~20 GB | 24+ GB |
| `mlx-community/Qwen3.6-35B-A3B-6bit` | ~30 GB | 40+ GB |
| `mlx-community/Qwen3.6-35B-A3B-8bit` | ~40 GB | 48+ GB |
| `mlx-community/Qwen3.6-35B-A3B-bf16` | ~70 GB | 96+ GB |

### MLX Benchmarks (Apple Silicon, 2026-04-20)

Tested with `test_chat.py` (3 runs + warmup), `mlx-openai-server` 1.7.1, `mlx-lm` 0.31.2. tok/s counts all generated tokens (think + answer) over full wall time.

| Run | TTFT | tok/s | Think time |
|-----|-----:|------:|-----------:|
| 1 | 25.4s | 80.1 | 25.2s |
| 2 | 19.1s | 79.9 | 18.9s |
| 3 | 22.6s | 79.7 | 22.4s |
| **avg** | **22.4s** | **79.9** | **22.2s** |

Think time dominates TTFT — the model reasons extensively before answering. Decode throughput is consistent at ~80 tok/s including think tokens.

## Compose variants

| File | Engine | Quant | Hardware | Notes |
|------|--------|-------|----------|-------|
| `docker-compose.vllm-35b-fp8-rtx.yml` | vLLM | FP8 | RTX PRO 6000 | Primary config, 262K context |
| `docker-compose.vllm-35b-fp8-textonly-rtx.yml` | vLLM | FP8 | RTX PRO 6000 | FP8 + vision disabled, fastest vLLM variant |
| `docker-compose.vllm-35b-text-only.yaml` | vLLM | BF16 | Any | Vision disabled, saves ~2-3 GB |
| `docker-compose.vllm-35b-1m.yaml` | vLLM | BF16 | Any | ~1M context via YaRN, util 0.95 |
| `docker-compose.sglang-35b-fp8.yaml` | SGLang | FP8 | Any | MTP speculative decoding |
| `docker-compose.llama-35b-devfix-rtx.yml` | llama.cpp | Q4_K_XL | RTX PRO 6000 | GGUF fallback (~22 GB) |

## Benchmarks (RTX PRO 6000, 96 GB)

Tested 2026-04-16 with `test_chat.py` (default MoE prompt) and `test_tools.py` (4 scenarios).

### Performance

| Config | tok/s | TTFT | Think time | Engine |
|--------|------:|-----:|-----------:|--------|
| **SGLang FP8** | **204.5** | **9.2s** | 8.4s | SGLang + MTP/NEXTN speculative decoding |
| **llama.cpp Q4_K_XL** | **205.3** | **9.2s** | 8.9s | llama.cpp GGUF, single slot |
| **vLLM FP8 text-only** | **203.2** | **7.3s** | 7.3s | vLLM v0.19.0, FP8 + vision disabled |
| **vLLM FP8** | 92.0 | 24.7s | 7.8s | vLLM v0.19.0, prefix caching |
| **vLLM BF16 text-only** | 86.6 | 27.0s | 10.2s | vLLM v0.19.0, vision disabled |
| **vLLM BF16 1M** | — | — | — | OOM: BF16 weights (66 GB) leave no room for 1M KV cache on 96 GB |

SGLang's MTP speculative decoding and llama.cpp's Q4 quantization both achieve ~200+ tok/s.
vLLM without speculative decoding sits at ~90 tok/s. The 1M context BF16 config does not fit
on the RTX PRO 6000 — use FP8 or a larger GPU.

### Tool calling

| Config | Single | Parallel (2) | Chained | Multi-parallel (3) | Total |
|--------|:------:|:------------:|:-------:|:------------------:|:-----:|
| **vLLM FP8** | pass | **fail** | pass | pass | 3/4 |
| **vLLM BF16 text-only** | pass | pass | pass | pass | 4/4 |
| **SGLang FP8** | pass | pass | pass | pass | 4/4 |
| **llama.cpp Q4_K_XL** | pass | pass (1 call) | pass | pass (1 call) | 4/4* |

\* llama.cpp passes all scenarios but does not emit parallel tool calls — it dispatches
sequentially (1 call at a time), so scenarios 2 and 4 work but with warnings.

vLLM FP8 fails scenario 2 (parallel 2-city weather) by returning JSON text instead of
structured tool calls. BF16 does not have this issue. This appears to be model-level
behavior with FP8 quantization, not a vLLM bug.

### Startup time

| Config | Cold start | Notes |
|--------|:----------:|-------|
| vLLM FP8 | ~3 min | FP8 weights ~35 GB, CUDA graph capture |
| vLLM BF16 text-only | ~12 min | BF16 weights ~70 GB + Triton MoE compilation (21s torch.compile) |
| SGLang FP8 | ~3 min | FP8 weights, RadixAttention init |
| llama.cpp Q4_K_XL | ~5 min | GGUF download (~22 GB) on first run, then ~1 min |

### Known issues

- **vLLM 1M context (BF16)**: Crashes in a restart loop on 96 GB GPU. BF16 weights consume
  66 GB, leaving insufficient VRAM for 1M-token KV cache. Needs FP8 weights or >128 GB VRAM.
- **SGLang + MTP + DeltaNet**: Requires `--mamba-scheduler-strategy extra_buffer` and
  `SGLANG_ENABLE_SPEC_V2=1` to avoid radix cache conflict with speculative decoding.
- **FP8 KV cache**: Omitted on all configs — DeltaNet linear attention state produces
  silent corruption with FP8 KV quantization (vllm-project/vllm#26646).
- **llama.cpp parallel tool calls**: Does not support emitting multiple tool calls in a
  single response; dispatches them sequentially.

## Testing

Run the shared test scripts to verify the endpoint:

```bash
# Chat benchmark — measures TTFT and tok/s
python ../shared/test_chat.py --model qwen3.6-35b

# Tool-calling integration test — single, parallel, chained, multi-parallel
python ../shared/test_tools.py --model qwen3.6-35b
```

## Requirements

- **vLLM**: v0.19.0+ (`vllm/vllm-openai:v0.19.0-cu130`)
- **SGLang**: v0.5.10+ (`lmsysorg/sglang:latest`)
- **llama.cpp**: latest `server-cuda` image

---

# Qwen3.6-27B

[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) — 27B dense model with the same hybrid attention architecture as the 35B MoE. First open-weight single-dense variant of the Qwen3.6 family.

## Architecture

- **Dense 27B**: all 27B parameters active per token (vs 3B for 35B-A3B MoE)
- **Hybrid attention**: 64 layers = 16 × (3×Gated DeltaNet + 1×full Gated Attention) — 48 DeltaNet + 16 full
- **Multimodal**: text, image, and video input (vision encoder optional)
- **Context**: 262K native, 1M via YaRN
- **Multi-Token Prediction**: native MTP for speculative decoding
- **License**: Apache 2.0

## Quick start

```bash
# RTX PRO 6000 (96 GB) — primary config
docker compose -f docker-compose.vllm-27b-fp8-rtx.yml up -d
```

API endpoint: `http://localhost:11436/v1`, model name: `qwen3.6-27b`

## Compose variants

| File | Engine | Quant | Hardware | Notes |
|------|--------|-------|----------|-------|
| `docker-compose.vllm-27b-fp8-rtx.yml` | vLLM | FP8 | RTX PRO 6000 | Primary config, 262K context |
| `docker-compose.vllm-27b-fp8-textonly-rtx.yml` | vLLM | FP8 | RTX PRO 6000 | FP8 + vision disabled |
| `docker-compose.vllm-27b-text-only.yaml` | vLLM | BF16 | Any | Vision disabled, ~54 GB BF16 weights |
| `docker-compose.sglang-27b-fp8.yaml` | SGLang | FP8 | Any | MTP speculative decoding |

No llama.cpp config — Qwen3.6-27B is very new and no community GGUF quants are available yet (as of 2026-04-22). Check `unsloth/Qwen3.6-27B-GGUF` or `bartowski/Qwen_Qwen3.6-27B-GGUF` for later availability.

## Benchmarks (RTX PRO 6000, 96 GB)

Tested 2026-04-22 with `test_chat.py` (default MoE prompt).

### Performance

| Config | tok/s | TTFT | Think time | Engine |
|--------|------:|-----:|-----------:|--------|
| **vLLM FP8 text-only** | 47.0 | 35.4s | 35.3s | vLLM v0.19.0, FP8 + vision disabled |
| **vLLM FP8** | 47.0 | 41.1s | 41.0s | vLLM v0.19.0, FP8 multimodal |
| **vLLM BF16 text-only** | 28.3 | 64.9s | 64.8s | vLLM v0.19.0, BF16 + vision disabled |
| **SGLang FP8 + MTP** | — | — | — | Config present but NEXTN spec decoding produces garbage (EAGLE fallback); fix pending |

Dense 27B decode is bandwidth-bound on all 27B active params (vs 3B for 35B-MoE):
~1.6 TB/s ÷ 27 GB ≈ 60 tok/s theoretical peak at FP8. Measured 47 tok/s is ~78%
efficiency, typical for FP8 tensor-core kernels on Blackwell SM_120.

BF16 weights (~54 GB) are 2× the size, so decode drops to 28 tok/s — exactly
halved, confirming the bandwidth-bound profile.

The vision encoder adds negligible overhead for text-only inference (both FP8
variants measured identical throughput).

### Startup time

| Config | Cold start | Notes |
|--------|:----------:|-------|
| vLLM FP8 | ~3.5 min | FP8 weights ~27 GB, CUDA graph capture |
| vLLM FP8 text-only | ~3.5 min | Same as FP8, vision encoder still loads |
| SGLang FP8 | ~2 min | Fast startup; MTP draft heads initialize quickly |
| vLLM BF16 text-only | ~12 min | BF16 weights ~54 GB + Triton compilation |

### Known issues

- **No GGUF**: community quants not yet published for 27B.
- **FP8 KV cache**: Omitted on all configs — DeltaNet linear attention state
  produces silent corruption with FP8 KV quantization (vllm-project/vllm#26646).

## Testing

```bash
# Chat benchmark
python ../shared/test_chat.py --base-url http://localhost:11436/v1 --model qwen3.6-27b

# Tool-calling integration test
python ../shared/test_tools.py --base-url http://localhost:11436/v1 --model qwen3.6-27b
```

## When to pick 27B vs 35B-A3B

- **35B-A3B (MoE)**: ~4× faster decode (200+ tok/s vs 47 tok/s). Best for high-throughput, multi-user, or latency-sensitive serving.
- **27B (dense)**: stronger per-token reasoning and generation quality. Best for single-user agentic tasks where quality matters more than speed.
