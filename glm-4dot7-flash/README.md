# GLM-4.7-Flash Server Setup

Local server deployment for GLM-4.7-Flash model with two backend options:
- **llama.cpp** - GGUF quantized models (Q4/Q8), lower VRAM, simpler setup
- **vLLM** - FP8/BF16, better parallelization, higher throughput

## Overview

- **Model**: GLM-4.7-Flash
- **Architecture**: 30B MoE (uses ~3.6B active parameters)
- **Context Window**: Up to 202,752 tokens
- **Port**: 11346
- **API**: OpenAI-compatible

### Backend Comparison

| Aspect | llama.cpp | vLLM |
|--------|-----------|------|
| Quantization | Q4_K_XL (~18GB), Q8_K_XL (~33GB) | FP8-Dynamic (~18GB), BF16 (~60GB) |
| Parallelization | 4 fixed slots | Continuous batching |
| Throughput | ~40-80 tok/s | ~250+ tok/s |
| Concurrent requests | Limited | Scales with memory |
| Setup complexity | Simple | Requires more VRAM headroom |

## Quick Start

### 1. Download Model (~18GB)

```bash
./download_model.sh
```

This downloads the model to `models/` directory using HuggingFace CLI.

### 2. Start Server

```bash
./start_server.sh
```

Server will be available at `http://127.0.0.1:11346/v1`

### 3. Test Server

```bash
./test_server.sh
```

## Configuration Details

### Server Settings

- **Port**: 11346
- **Context Size**: 202,752 tokens (full)
- **Temperature**: 0.7 (optimized for tool-calling)
- **Top-p**: 1.0
- **Repeat Penalty**: 1.0 (critical - do not change)
- **Threads**: Auto-detect
- **Memory**: Fit-on mode enabled

### Tool-Calling Mode

The server is configured with tool-calling optimized parameters:
- `--temp 0.7 --top-p 1.0` (recommended for tool calling)
- For general use, change to: `--temp 1.0 --top-p 0.95`

## Client Usage

### Python Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11346/v1",
    api_key="sk-no-key-required"
)

response = client.chat.completions.create(
    model="unsloth/GLM-4.7-Flash",
    messages=[
        {"role": "user", "content": "What is 2+2?"}
    ]
)

print(response.choices[0].message.content)
```

### curl Example

```bash
curl http://127.0.0.1:11346/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/GLM-4.7-Flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Docker Deployment

### Build Image

```bash
docker build -t glm-4-7-flash .
```

### Run with Docker Compose

```bash
# First, download model
docker run -v $(pwd)/models:/app/models glm-4-7-flash ./download_model.py

# Start server
docker compose up -d
```

Server will be available at `http://127.0.0.1:11346/v1`

## vLLM Deployment (Alternative)

vLLM offers better parallelization and throughput for concurrent requests.

### Start with vLLM

```bash
# FP8 quantized (recommended - lower VRAM, 200K context)
docker compose -f docker-compose.vllm.yml --profile fp8 up -d

# BF16 full precision (higher quality, more VRAM, 131K context)
docker compose -f docker-compose.vllm.yml --profile bf16 up -d
```

### Test vLLM Server

```bash
./test_vllm.sh
```

### vLLM Configuration

| Profile | Model | VRAM | Context |
|---------|-------|------|---------|
| fp8 | GLM-4.7-Flash-FP8-Dynamic | ~30GB | 200K |
| bf16 | GLM-4.7-Flash | ~70GB | 131K |

Both profiles use the same port (11346) - only one can run at a time.

### Stop vLLM

```bash
docker compose -f docker-compose.vllm.yml --profile fp8 down
```

## VRAM Requirements

### llama.cpp
| Model | Base | With 202K Context |
|-------|------|-------------------|
| Q4_K_XL | ~18GB | ~38GB |
| Q8_K_XL | ~33GB | ~58GB |

### vLLM
| Profile | Estimated VRAM |
|---------|----------------|
| FP8 (200K ctx) | ~30GB |
| BF16 (131K ctx) | ~70GB |

Check current usage:
```bash
nvidia-smi
```

## Files

### llama.cpp Backend
- `build.sh` - Build llama.cpp with CUDA support
- `download_model.sh` - Download GGUF models using HuggingFace CLI
- `start_server.sh` - Start llama-server (Q4)
- `start_server_q8.sh` - Start llama-server (Q8)
- `test_server.sh` - Test llama.cpp server
- `Dockerfile` - Docker container for llama.cpp
- `docker-compose.yml` - Docker Compose (profiles: q4, q8)

### vLLM Backend
- `docker-compose.vllm.yml` - Docker Compose for vLLM (profiles: fp8, bf16)
- `test_vllm.sh` - Test vLLM server

## Blackwell GPU Compatibility

This setup includes a fix for NVIDIA Blackwell GPUs (compute capability 12.0):

### The Problem
- Blackwell has compute capability 12.0 (compute_120a)
- Current NVCC doesn't support compute capability 12.0 yet
- CMake auto-detection fails

### The Solution
We manually specify older CUDA architectures in `build.sh`:

```bash
cmake llama.cpp -B llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
```

**CUDA Architectures**:
- 80 = Ampere (RTX 3000 series, A100)
- 86 = Ampere (RTX 3090, A6000)
- 89 = Ada Lovelace (RTX 4000 series, L40)
- 90 = Hopper (H100)

**Trade-offs**:
- ✅ Code compiles and runs on Blackwell
- ✅ CUDA acceleration works
- ⚠️ No Blackwell-specific optimizations (uses backward-compatible code paths)

**Future**: Update CUDA toolkit when it supports compute capability 12.0

## References

- [Unsloth Documentation](https://unsloth.ai/docs/models/glm-4.7-flash)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- Model: [unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)

## Troubleshooting

**Server won't start**:
- Check model is downloaded: `ls -lh models/`
- Check llama-server is built: `ls -lh llama.cpp/build/bin/llama-server`

**High VRAM usage**:
- Reduce context size: change `--ctx-size 202752` to `--ctx-size 16384` in `start_server.sh`

**Slow responses**:
- Check GPU utilization: `nvidia-smi`
- Ensure CUDA is enabled during build