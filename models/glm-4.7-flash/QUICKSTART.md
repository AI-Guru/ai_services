# GLM-4.7-Flash Quick Start Guide

## Current Status

✅ **Build Complete** - llama.cpp with CUDA support built successfully
✅ **Scripts Ready** - All deployment scripts created
⏳ **Model Download** - Needs to be downloaded (~18GB)
⏳ **Server** - Ready to start once model is downloaded

## Next Steps

### 1. Download the Model (~18GB, may take 10-30 minutes)

```bash
cd /home/hordak/Development/ai_services/glm-4dot7-flash
./download_model.sh
```

Uses HuggingFace CLI (`hf` command) to download the model.

### 2. Start the Server

```bash
./start_server.sh
```

The server will start on **port 11346** with:
- Full 202K context window support
- Tool-calling optimized configuration
- OpenAI-compatible API at `http://127.0.0.1:11346/v1`

### 3. Test the Server

In a new terminal:

```bash
./test_server.sh
```

This will:
- Test basic chat completion with curl
- Test tool-calling capability
- Display VRAM usage

## Configuration Summary

| Setting | Value |
|---------|-------|
| Port | 11346 |
| Context Size | 202,752 tokens (full) |
| Temperature | 0.7 (tool-calling mode) |
| Top-P | 1.0 |
| Model Size | ~18GB (Q4_K_XL quantization) |
| VRAM Required | ~18-24GB |

## Files Created

- `build.sh` - Builds llama.cpp (✅ completed)
- `download_model.sh` - Downloads model using `hf` CLI
- `start_server.sh` - Starts the server
- `test_server.sh` - Tests server with curl
- `Dockerfile` - For Docker deployment
- `docker-compose.yml` - Docker Compose config
- `README.md` - Full documentation

## Example Usage

Once the server is running:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11346/v1",
    api_key="sk-no-key-required"
)

response = client.chat.completions.create(
    model="unsloth/GLM-4.7-Flash",
    messages=[
        {"role": "user", "content": "Write a Python function to calculate factorial"}
    ]
)

print(response.choices[0].message.content)
```

## Docker Alternative

If you prefer Docker:

```bash
# Build image
docker build -t glm-4-7-flash .

# Download model
mkdir -p models
docker run -v $(pwd)/models:/app/models glm-4-7-flash ./download_model.py

# Start with Docker Compose
docker compose up -d
```

## Monitoring

Watch GPU usage:
```bash
watch -n 1 nvidia-smi
```

Check server logs:
```bash
# If running manually: see terminal output
# If running with Docker: docker compose logs -f
```

## Hardware Verified

- GPU: NVIDIA RTX PRO 6000 Blackwell (97.8GB VRAM) ✅
- CUDA: 12.4 ✅
- Compatibility: Blackwell-specific build configuration applied ✅

## Troubleshooting

**If download fails:**
- Check internet connection
- Ensure `pip install huggingface_hub hf_transfer` was successful
- Try running download script with `python3 ./download_model.py`

**If server won't start:**
- Verify model downloaded: `ls -lh models/`
- Check port not in use: `lsof -i :11346`

**For more details, see README.md**
