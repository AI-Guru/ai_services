# GLM-4.7-Flash Server Setup

Local server deployment for GLM-4.7-Flash model using llama.cpp with CUDA acceleration.

## Overview

- **Model**: GLM-4.7-Flash (Q4_K_XL quantization)
- **Architecture**: 30B MoE (uses ~3.6B active parameters)
- **Context Window**: 202,752 tokens (full support)
- **Port**: 11346
- **API**: OpenAI-compatible
- **Backend**: llama.cpp with CUDA

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

## VRAM Requirements

- **Quantized (Q4_K_XL)**: ~18GB VRAM
- **Full Context (202K)**: Additional memory based on usage
- **Recommended**: 24GB+ VRAM

Check current usage:
```bash
nvidia-smi
```

## Files

- `build.sh` - Build llama.cpp with CUDA support
- `download_model.sh` - Download GLM-4.7-Flash model using HuggingFace CLI
- `start_server.sh` - Start llama-server
- `test_server.sh` - Test server with curl (includes VRAM check)
- `Dockerfile` - Docker container setup
- `docker-compose.yml` - Docker Compose configuration

**Optional Python scripts** (if you prefer Python):
- `download_model.py` - Python version of model downloader
- `test_client.py` - Python test client with OpenAI library

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