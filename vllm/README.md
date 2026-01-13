# vLLM Docker Setup for GPT-OSS

Docker Compose setup for running OpenAI's gpt-oss models with vLLM on NVIDIA GPUs.

## Quick Start

```bash
# 1. Prepare environment (downloads tiktoken encoding files)
./prepare.sh

# 2. Start the server
docker compose up -d

# 3. Watch logs (model downloads on first run - ~40GB for gpt-oss-20b)
docker compose logs -f vllm

# 4. Test when ready
curl http://localhost:11435/v1/models
```

The server will be available at `http://localhost:11435` with OpenAI-compatible API.

## vLLM vs Ollama: When to Use Which

### Use vLLM when:
- **High throughput needed**: 5-7x faster than Ollama (~250 tok/s vs ~40 tok/s for gpt-oss-20b)
- **Production deployments**: Optimized for serving at scale with batching and KV cache
- **OpenAI API compatibility required**: Drop-in replacement for OpenAI endpoints
- **Multiple concurrent requests**: Handles concurrent users efficiently with request batching
- **GPU memory is abundant**: Pre-allocates memory for optimal performance

### Use Ollama when:
- **Interactive development**: Quick model switching and experimentation
- **Low memory footprint**: Uses ~4GB vs ~42GB for the same model
- **Simple local testing**: Easy CLI interface, no API boilerplate needed
- **Running multiple models**: Small memory footprint allows many models loaded simultaneously
- **CPU inference**: Supports CPU-only inference (vLLM requires GPU)

### Summary
- **vLLM**: Production server - fast, optimized, but memory-hungry
- **Ollama**: Development tool - flexible, lightweight, interactive

Both can run simultaneously (Ollama on :11434, vLLM on :11435) for development + production workflows.

## Hardware Requirements

**gpt-oss-20b** (configured model):
- GPU: 48GB+ VRAM (RTX PRO 6000/H100/A100 80GB recommended)
- RAM: 32GB+ system memory
- Storage: 50GB for model weights

**gpt-oss-120b** (optional):
- GPU: 80GB+ VRAM required
- Change model in [docker-compose.yml:54](docker-compose.yml#L54)

## Configuration

### Optional: Hugging Face Token

Only needed if you encounter authentication errors:

```bash
# Create .env file
echo "HF_TOKEN=hf_your_token_here" > .env
```

### Memory Tuning

If you get OOM errors, reduce memory usage by editing the command in [docker-compose.yml:53-57](docker-compose.yml#L53-L57):

```yaml
command: >
  vllm serve openai/gpt-oss-20b
  --host 0.0.0.0
  --port 8000
  --gpu-memory-utilization 0.85
  --max-model-len 32768
```

## Usage

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11435/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7
)
print(response.choices[0].message.content)
```

### cURL

```bash
# Health check
curl http://localhost:11435/health

# Chat completion
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Testing & Benchmarking

### test.sh - Basic Performance Test

Tests single-request performance with long poem generation:

```bash
./test.sh
```

Measures tokens/second for a single request generating ~1500 tokens.

### compare.sh - Ollama vs vLLM Comparison

Compares performance between Ollama (port 11434) and vLLM (port 11435):

```bash
./compare.sh
```

Runs identical prompts on both servers and shows side-by-side metrics.

### stresstest.sh - Parallel Load Testing

Tests concurrent request handling to measure throughput under load:

```bash
# Default: 10 requests, 5 concurrent
./stresstest.sh

# Custom load
NUM_REQUESTS=50 CONCURRENCY=16 ./stresstest.sh
```

Recommended progression:
- Phase 1: `./stresstest.sh` (baseline)
- Phase 2: `NUM_REQUESTS=20 CONCURRENCY=10 ./stresstest.sh`
- Phase 3: `NUM_REQUESTS=50 CONCURRENCY=16 ./stresstest.sh` (LangGraph typical)
- Phase 4: `NUM_REQUESTS=100 CONCURRENCY=25 ./stresstest.sh` (production load)
- Phase 5: `NUM_REQUESTS=200 CONCURRENCY=50 ./stresstest.sh` (high concurrency)

See [stresstest.md](stresstest.md) for detailed benchmarking guide with LLMPerf integration.

## Docker Commands

```bash
# Start
docker compose up -d

# Stop
docker compose down

# View logs
docker compose logs -f vllm

# Restart after config changes
docker compose restart vllm

# Remove everything including cached models
docker compose down -v
```

## Troubleshooting

**Container won't start:**
```bash
# Check NVIDIA runtime
docker info | grep -i nvidia

# Reinstall if needed
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

**Out of memory:**
- Reduce `--gpu-memory-utilization` to 0.80 or lower
- Reduce `--max-model-len` to 32768 or 16384
- Check no other processes are using GPU: `nvidia-smi`