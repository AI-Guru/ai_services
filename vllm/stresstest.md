# Stress Testing vLLM and Ollama with LLMPerf

This guide shows how to benchmark and stress test both Ollama (port 11434) and vLLM (port 11435) using parallel concurrent requests.

## Prerequisites

Install llmperf:

```bash
git clone https://github.com/ray-project/llmperf.git
cd llmperf
pip install -e .
```

## Quick Start

### Test vLLM (Port 11435)

```bash
cd llmperf

export OPENAI_API_KEY="not-needed"
export OPENAI_API_BASE="http://localhost:11435/v1"

python token_benchmark_ray.py \
  --model "openai/gpt-oss-20b" \
  --mean-input-tokens 100 \
  --stddev-input-tokens 20 \
  --mean-output-tokens 200 \
  --stddev-output-tokens 20 \
  --max-num-completed-requests 50 \
  --timeout 600 \
  --num-concurrent-requests 10 \
  --results-dir "results_vllm" \
  --llm-api openai \
  --additional-sampling-params '{"temperature": 0.7}'
```

### Test Ollama (Port 11434)

Ollama doesn't use the OpenAI `/v1/` prefix, so we need to use a different approach. Use the custom script below.

## Test Scenarios

### 1. Low Concurrency (1-4 requests)
**Use case**: Single user, sequential workflow

```bash
# vLLM - Low concurrency
python token_benchmark_ray.py \
  --model "openai/gpt-oss-20b" \
  --mean-input-tokens 100 \
  --mean-output-tokens 200 \
  --max-num-completed-requests 20 \
  --num-concurrent-requests 1 \
  --results-dir "results_vllm_low" \
  --llm-api openai
```

**Expected results**:
- vLLM: ~250 tok/s per request
- Ollama: ~40 tok/s per request

### 2. Medium Concurrency (8-16 requests)
**Use case**: LangGraph parallel workflows, multi-agent systems

```bash
# vLLM - Medium concurrency
python token_benchmark_ray.py \
  --model "openai/gpt-oss-20b" \
  --mean-input-tokens 150 \
  --mean-output-tokens 300 \
  --max-num-completed-requests 100 \
  --num-concurrent-requests 8 \
  --results-dir "results_vllm_medium" \
  --llm-api openai
```

**Expected results**:
- vLLM: ~150-200 tok/s per request (benefits from batching)
- Ollama: Queues requests, ~40 tok/s (limited by OLLAMA_NUM_PARALLEL)

### 3. High Concurrency (20-32 requests)
**Use case**: Production API serving multiple users

```bash
# vLLM - High concurrency
python token_benchmark_ray.py \
  --model "openai/gpt-oss-20b" \
  --mean-input-tokens 200 \
  --mean-output-tokens 400 \
  --max-num-completed-requests 200 \
  --num-concurrent-requests 20 \
  --results-dir "results_vllm_high" \
  --llm-api openai
```

**Expected results**:
- vLLM: ~100-150 tok/s per request (your setup supports 28x concurrency)
- Ollama: Significant queuing, latency increases dramatically

### 4. Stress Test (Maximum Load)
**Use case**: Finding breaking points and maximum throughput

```bash
# vLLM - Stress test
python token_benchmark_ray.py \
  --model "openai/gpt-oss-20b" \
  --mean-input-tokens 500 \
  --mean-output-tokens 1000 \
  --max-num-completed-requests 500 \
  --num-concurrent-requests 32 \
  --timeout 1200 \
  --results-dir "results_vllm_stress" \
  --llm-api openai
```

## Custom Comparison Script

Since llmperf works best with OpenAI-compatible endpoints, here's a custom bash script for direct comparison:

```bash
#!/bin/bash
# vllm/stresstest.sh - Parallel load test for both servers

set -e

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
VLLM_URL="${VLLM_URL:-http://localhost:11435}"
NUM_REQUESTS="${NUM_REQUESTS:-10}"
CONCURRENCY="${CONCURRENCY:-5}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Parallel Stress Test: gpt-oss-20b"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Total requests: ${NUM_REQUESTS}"
echo "Concurrency: ${CONCURRENCY}"
echo ""

# Test vLLM
echo "Testing vLLM (${VLLM_URL})..."
START=$(date +%s.%N)

seq 1 ${NUM_REQUESTS} | xargs -P ${CONCURRENCY} -I {} bash -c "
    curl -s '${VLLM_URL}/v1/chat/completions' \
        -H 'Content-Type: application/json' \
        -d '{
            \"model\": \"openai/gpt-oss-20b\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Write a short poem about AI\"}],
            \"temperature\": 0.7,
            \"max_tokens\": 100
        }' > /dev/null
"

END=$(date +%s.%N)
VLLM_DURATION=$(echo "${END} - ${START}" | bc)
VLLM_RPS=$(echo "scale=2; ${NUM_REQUESTS} / ${VLLM_DURATION}" | bc)

echo "  Duration: ${VLLM_DURATION}s"
echo "  Requests/sec: ${VLLM_RPS}"
echo ""

# Test Ollama
echo "Testing Ollama (${OLLAMA_URL})..."
START=$(date +%s.%N)

seq 1 ${NUM_REQUESTS} | xargs -P ${CONCURRENCY} -I {} bash -c "
    curl -s '${OLLAMA_URL}/api/generate' \
        -d '{
            \"model\": \"gpt-oss:20b\",
            \"prompt\": \"Write a short poem about AI\",
            \"stream\": false,
            \"options\": {\"temperature\": 0.7, \"num_predict\": 100}
        }' > /dev/null
"

END=$(date +%s.%N)
OLLAMA_DURATION=$(echo "${END} - ${START}" | bc)
OLLAMA_RPS=$(echo "scale=2; ${NUM_REQUESTS} / ${OLLAMA_DURATION}" | bc)

echo "  Duration: ${OLLAMA_DURATION}s"
echo "  Requests/sec: ${OLLAMA_RPS}"
echo ""

# Comparison
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-20s %15s %15s\n" "Metric" "vLLM" "Ollama"
echo "────────────────────────────────────────────────────"
printf "%-20s %15s %15s\n" "Duration (s)" "${VLLM_DURATION}" "${OLLAMA_DURATION}"
printf "%-20s %15s %15s\n" "Requests/sec" "${VLLM_RPS}" "${OLLAMA_RPS}"

SPEEDUP=$(echo "scale=2; ${OLLAMA_DURATION} / ${VLLM_DURATION}" | bc)
echo ""
echo "vLLM is ${SPEEDUP}x faster than Ollama for concurrent requests"
```

Save as `stresstest.sh` and run:

```bash
chmod +x stresstest.sh

# Light test
NUM_REQUESTS=10 CONCURRENCY=5 ./stresstest.sh

# Medium test
NUM_REQUESTS=50 CONCURRENCY=10 ./stresstest.sh

# Heavy test
NUM_REQUESTS=100 CONCURRENCY=20 ./stresstest.sh
```

## Understanding Results

### Key Metrics

1. **Throughput (tokens/sec)**: How fast the model generates text
2. **Requests per second**: How many concurrent requests complete per second
3. **P50/P99 Latency**: 50th and 99th percentile response times
4. **Time to First Token (TTFT)**: Latency before generation starts
5. **Inter-Token Latency (ITL)**: Time between generated tokens

### LLMPerf Output

Results are saved in the specified `--results-dir`:
- `summary.json`: Aggregate metrics across all requests
- Individual request files: Per-request detailed metrics

Example summary metrics:
```json
{
  "mean_ttft_ms": 45.2,
  "median_ttft_ms": 43.1,
  "p99_ttft_ms": 78.5,
  "mean_itl_ms": 3.8,
  "median_itl_ms": 3.5,
  "p99_itl_ms": 8.2,
  "mean_output_throughput": 248.5,
  "completed_requests": 100
}
```

## Interpreting Results for Your Setup

### vLLM (50% GPU utilization, 26GB KV cache)

- **1-4 concurrent**: ~250 tok/s each (no batching penalty)
- **8-16 concurrent**: ~150-200 tok/s each (efficient batching)
- **20-28 concurrent**: ~100-150 tok/s each (at KV cache limit)
- **>28 concurrent**: Requests queue, latency increases

### Ollama (default config)

- **1-4 concurrent**: ~40 tok/s each (OLLAMA_NUM_PARALLEL=4 default)
- **>4 concurrent**: Requests queue, processes 4 at a time
- Even with OLLAMA_NUM_PARALLEL=32, performance plateaus at ~22 req/s

## Advanced: Monitoring During Tests

### Monitor GPU usage

```bash
watch -n 1 nvidia-smi
```

### Monitor vLLM metrics

```bash
curl http://localhost:11435/metrics
```

Key metrics to watch:
- `vllm:num_requests_running`: Active requests being processed
- `vllm:num_requests_waiting`: Queued requests
- `vllm:gpu_cache_usage_perc`: KV cache utilization
- `vllm:time_to_first_token_seconds`: TTFT distribution

### Monitor Docker container stats

```bash
docker stats vllm-gpt-oss
```

## Tuning for Your Workload

### If you need more concurrent capacity:

Increase GPU memory allocation in [docker-compose.yml](docker-compose.yml#L61):
```yaml
--gpu-memory-utilization
- "0.70"  # Increase from 0.50
```

This will give you more KV cache (35GB instead of 26GB) and higher concurrency (~40x instead of 28x).

### If you need lower latency:

Reduce concurrency and increase GPU allocation:
```yaml
--gpu-memory-utilization
- "0.40"  # Reduce to 0.40 for faster single requests
```

Lower memory means faster context switching but lower max concurrency.

## LangGraph-Specific Testing

For testing LangGraph parallel workflows, create a Python test:

```python
# test_langgraph_parallel.py
import asyncio
import time
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:11435/v1",
    api_key="not-needed"
)

async def parallel_agent(agent_id: int):
    start = time.time()
    response = await client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{
            "role": "user",
            "content": f"Agent {agent_id}: Analyze topic {agent_id}"
        }],
        temperature=0.7,
        max_tokens=200
    )
    duration = time.time() - start
    return agent_id, duration, len(response.choices[0].message.content)

async def test_parallel_agents(num_agents: int):
    print(f"Running {num_agents} parallel agents...")
    start = time.time()

    tasks = [parallel_agent(i) for i in range(num_agents)]
    results = await asyncio.gather(*tasks)

    total_duration = time.time() - start

    print(f"\nCompleted {num_agents} agents in {total_duration:.2f}s")
    for agent_id, duration, length in results:
        print(f"  Agent {agent_id}: {duration:.2f}s, {length} chars")

    print(f"\nAverage per agent: {sum(r[1] for r in results)/len(results):.2f}s")
    print(f"Effective speedup: {sum(r[1] for r in results)/total_duration:.2f}x")

# Test with increasing parallelism
for n in [1, 4, 8, 16]:
    asyncio.run(test_parallel_agents(n))
    print("\n" + "="*60 + "\n")
```

Run with:
```bash
python test_langgraph_parallel.py
```

This simulates realistic LangGraph parallel node execution patterns.

## Expected Performance Summary

| Concurrent Requests | vLLM (tok/s) | Ollama (tok/s) | vLLM Advantage |
|---------------------|--------------|----------------|----------------|
| 1                   | 250          | 40             | 6.2x           |
| 4                   | 200          | 40             | 5.0x           |
| 8                   | 150          | 35*            | 4.3x           |
| 16                  | 100          | 25*            | 4.0x           |
| 28                  | 75           | 15*            | 5.0x           |

\* Ollama performance degrades with queuing beyond OLLAMA_NUM_PARALLEL

## Troubleshooting

**Error: "Connection refused"**
- Check servers are running: `docker ps` and `ollama list`
- Verify ports: `curl http://localhost:11435/health` and `curl http://localhost:11434/api/tags`

**Error: "Max concurrency exceeded"**
- Your KV cache is full (28x limit reached)
- Reduce `--num-concurrent-requests` or increase GPU memory

**Slow performance**
- Check GPU isn't running other models: `nvidia-smi`
- Verify no other processes using VRAM
- Consider reducing `--mean-output-tokens` for shorter generations

**llmperf crashes**
- Increase `--timeout` for longer generations
- Reduce `--num-concurrent-requests`
- Check Ray cluster logs: `ray status`
