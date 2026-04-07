# Qwen3Guard — Safety Classification Models

Qwen3Guard-Gen models are generative safety classifiers that categorize prompts and
responses into **Safe**, **Controversial**, or **Unsafe** across 119 languages.
They cover categories including Violent, Sexual Content, PII, Suicide & Self-Harm,
Jailbreak, and more.

Source: https://github.com/QwenLM/Qwen3Guard

## Available Configurations

RTX PRO 6000 Blackwell (96 GB). Two backends: **vLLM v0.17.0** (bf16) and
**llama.cpp** (Q4_K_M GGUF, much lower VRAM).

### vLLM (bf16)

| Model | Compose File | Port | gpu-mem-util | Context |
|-------|-------------|------|-------------|---------|
| Gen-0.6B | `docker-compose.vllm-gen-0.6b-rtx.yml` | 11460 | 0.05 | 8192 |
| Gen-4B | `docker-compose.vllm-gen-4b-rtx.yml` | 11461 | 0.12 | 8192 |
| Gen-8B | `docker-compose.vllm-gen-8b-rtx.yml` | 11462 | 0.22 | 8192 |

### llama.cpp (Q4_K_M GGUF)

| Model | Compose File | Port | Context |
|-------|-------------|------|---------|
| Gen-0.6B | `docker-compose.llama-gen-0.6b-rtx.yml` | 11460 | 8192 |
| Gen-4B | `docker-compose.llama-gen-4b-rtx.yml` | 11461 | 8192 |
| Gen-8B | `docker-compose.llama-gen-8b-rtx.yml` | 11462 | 8192 |

## VRAM Requirements

| Model | Weights (bf16) | KV Cache (8192 ctx) | Overhead | Total |
|-------|---------------|--------------------|---------:|------:|
| Gen-0.6B | 1.2 GB | 0.9 GB | ~1.0 GB | ~3.1 GB |
| Gen-4B | 8.0 GB | 1.1 GB | ~1.5 GB | ~10.6 GB |
| Gen-8B | 16.0 GB | 1.1 GB | ~1.5 GB | ~18.6 GB |

Low `gpu-memory-utilization` values allow these guard models to run alongside
a large primary model on the same GPU.

## Benchmark Results

Tested with `test_chat.py --no-think --runs 3 --warmup --max-tokens 256`
on RTX PRO 6000 Blackwell (96 GB). Default prompt (MoE vs dense explanation).

### vLLM (bf16)

| Model | Avg TTFT | Avg tok/s | Tokens |
|-------|---------|----------|--------|
| Gen-0.6B | 12 ms | 395.9 | 8 |
| Gen-4B | 18 ms | 131.6 | 8 |
| Gen-8B | 25 ms | 80.7 | 8 |

### llama.cpp (Q4_K_M GGUF)

| Model | Avg TTFT | Avg tok/s | Tokens |
|-------|---------|----------|--------|
| Gen-0.6B | 7 ms | 409.8 | 8 |
| Gen-4B | 12 ms | 222.0 | 8 |
| Gen-8B | 14 ms | 166.9 | 8 |

llama.cpp is faster across the board thanks to Q4_K_M quantization reducing
memory bandwidth requirements. The difference is most pronounced on the larger
models (4B: 1.7x, 8B: 2.1x faster than vLLM bf16).

All models produce short structured output (~8 tokens: `Safety: Safe\nCategories: None`),
so throughput numbers reflect per-request latency rather than sustained generation speed.

## Usage

```bash
# Start a guard model
docker compose -f docker-compose.vllm-gen-4b-rtx.yml up -d

# Test it
python ../shared/test_chat.py \
  --base-url http://localhost:11461/v1 \
  --model qwen3guard-gen-4b \
  --no-think --runs 3 --warmup

# Stop
docker compose -f docker-compose.vllm-gen-4b-rtx.yml down
```

## Stream Models (Not Supported)

The Qwen3Guard-Stream models (0.6B, 4B, 8B) use the `Qwen3ForGuardModel`
architecture which is not supported by vLLM v0.17.0. Only the Gen models
(standard `Qwen3ForCausalLM`) are deployable with vLLM at this time.
