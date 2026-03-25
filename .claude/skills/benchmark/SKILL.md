---
name: benchmark
description: Benchmark a running LLM inference server (vLLM, llama.cpp, SGLang). Measures throughput (tok/s), latency (TTFT), and concurrency scaling across realistic workload scenarios.
argument-hint: "[target] [options]"
---

# LLM Inference Benchmark

Benchmark a running inference server with realistic workload scenarios.

## Usage

```
/benchmark                                    # Auto-detect running server
/benchmark http://localhost:11435/v1           # Specific endpoint
/benchmark http://localhost:11440/v1 --think   # With thinking enabled
/benchmark guidellm http://localhost:11435/v1  # Full GuideLLM sweep
```

## What to do

### Step 1 — Detect the target

If `$ARGUMENTS` includes a URL, use that. Otherwise auto-detect by probing common ports:

| Port | Typical use |
|------|-------------|
| 11430–11433 | Small models (0.8B–9B) |
| 11435 | Primary model (35B) |
| 11436 | Secondary model (27B) |
| 11437 | Large model (122B) |
| 11438 | Coder model |
| 11440 | Nemotron models |

```bash
curl -sf http://localhost:PORT/health
curl -sf http://localhost:PORT/v1/models
```

Identify the model name from `/v1/models` response.

### Step 2 — Choose benchmark mode

**Default: `test_scenarios.py`** (quick, 5 scenarios, ~5 min)

Run `models/shared/test_scenarios.py` which tests 5 realistic workload scenarios:

| Scenario | Input tokens | Output tokens | Simulates |
|----------|-------------|---------------|-----------|
| Chat | ~55 | ~300 | Short conversational Q&A |
| RAG | ~775 | ~300 | Retrieval with 4 doc chunks |
| Codegen | ~150 | ~1000 | File context to full function |
| Summarization | ~577 | ~200 | Long document to bullet points |
| Agentic | ~895 | ~600 | Tool-use agent with history |

```bash
python models/shared/test_scenarios.py \
  --base-url URL \
  --model MODEL_NAME \
  --runs 3 \
  --warmup
```

By default `--no-think` is on (skips `<think>` blocks). Add `--think` to enable reasoning.

**If `guidellm` is specified: full GuideLLM sweep** (~20 min per scenario)

This tests concurrency scaling from single-user to saturation using `guidellm benchmark` with sweep profile. Use this when you need to know how many concurrent users a GPU can support.

```bash
guidellm benchmark \
  --target TARGET_URL \
  --model MODEL_NAME \
  --processor HF_MODEL_ID \
  --profile sweep \
  --data "prompt_tokens=PT,prompt_tokens_stdev=PS,prompt_tokens_min=PMIN,prompt_tokens_max=PMAX,output_tokens=OT,output_tokens_stdev=OS,output_tokens_min=OMIN,output_tokens_max=OMAX" \
  --max-seconds 120 \
  --warmup 0.1 \
  --output-dir OUTPUT_DIR \
  --outputs json,csv
```

GuideLLM scenario parameters:

| Scenario | prompt_tokens (stdev) | output_tokens (stdev) |
|----------|----------------------|----------------------|
| chat | 2000 (500) | 300 (75) |
| rag | 8000 (1500) | 256 (64) |
| agentic | 16000 (3000) | 800 (200) |
| codegen | 4000 (1000) | 1500 (400) |
| summarization | 12000 (2000) | 300 (75) |

Set min/max to ~50%/150% of the mean for each.

**IMPORTANT**: Do NOT pipe guidellm output through `head`, `tail`, or `tee` — it causes SIGPIPE deadlocks. Always run with `2>&1` and no pipe truncation.

### Step 3 — Run and report

1. Verify server health first
2. Run the benchmark (background for guidellm, foreground for test_scenarios.py)
3. Report results in a markdown table with:
   - **Per-scenario tok/s** (chat, rag, codegen, summarization, agentic)
   - **Overall average tok/s**
   - **TTFT** per scenario
   - For GuideLLM: **peak concurrency**, **peak total tok/s**, **latency at peak**
4. Compare against known baselines if available (check README.md in the model directory)

### Step 4 — Interpret results

Provide practical guidance:
- How many concurrent users can this setup support?
- What's the bottleneck (decode speed, KV cache, prefill)?
- How does it compare to other models/quantizations we've tested?
- Is the model suitable for the intended use case (chat, coding, agentic)?

## Reference: known baselines on RTX PRO 6000

| Model | Backend | Overall tok/s |
|---|---|---|
| Nemotron Nano 4B | llama.cpp Q4_K_M | 299 |
| Nemotron Cascade-2 30B | vLLM AWQ | 273 |
| Nemotron Cascade-2 30B | llama.cpp Q4_K_M | 250 |
| Qwen3.5-35B-A3B | llama.cpp Q4_K_XL | 194 |
| Qwen3-Coder-Next | llama.cpp Q4_K_XL | 164 |
| Qwen3.5-35B-A3B | vLLM FP8 | 146 |
| Nemotron Super-120B | vLLM NVFP4 | 80 |

$ARGUMENTS
