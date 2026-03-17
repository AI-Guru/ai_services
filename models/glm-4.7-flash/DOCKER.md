# Docker Deployment Guide

## Quick Start

### 1. Build the Image (One-time, ~5-10 minutes)

```bash
docker compose build
```

### 2. Download Models (One-time, ~20-40 minutes)

```bash
docker compose run --rm glm-q4 ./download_model.sh
```

This downloads both Q4 (~18GB) and Q8 (~35GB) models to `./models/` directory.

### 3. Start Server

**Select profile for Q4 (4-bit, faster) OR Q8 (8-bit, higher quality):**

```bash
# Q4 model
docker compose --profile q4 up -d

# OR Q8 model
docker compose --profile q8 up -d
```

Both use port 11346 - only one can run at a time.

## Profiles

| Profile | Model | Port | VRAM | Quality |
|---------|-------|------|------|---------|
| q4 | Q4_K_XL | 11346 | ~20-25GB | Fast |
| q8 | Q8_K_XL | 11346 | ~40-45GB | Best |

## Common Commands

### View Logs
```bash
docker compose logs -f
```

### Stop Server
```bash
docker compose down
```

### Restart Server
```bash
# For Q4
docker compose --profile q4 restart

# For Q8
docker compose --profile q8 restart
```

### Switch Models
```bash
# Stop current
docker compose down

# Start different model
docker compose --profile q8 up -d
```

### Check VRAM Usage
```bash
docker exec glm-4-7-flash-server nvidia-smi
```

### Run Tests
```bash
docker exec glm-4-7-flash-server ./test_server.sh
```

## OpenCode Configuration

Same configuration works for both Q4 and Q8 (both use port 11346):

```json
{
  "baseURL": "http://hordak:11346/v1"
}
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs glm-q4

# Verify model is downloaded
ls -lh models/

# Rebuild image
docker compose build --no-cache
```

### GPU not detected
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu24.04 nvidia-smi

# Ensure nvidia-container-toolkit is installed
```

### Model not found
```bash
# Re-download models
docker compose run --rm glm-q4 ./download_model.sh
```

### Port already in use
```bash
# Check what's using the port
sudo lsof -i :11346
sudo lsof -i :11347

# Stop conflicting service or change port in docker-compose.yml
```

## Configuration

All server settings are in:
- Q4: [start_server.sh](start_server.sh)
- Q8: [start_server_q8.sh](start_server_q8.sh)

Both configured with:
- Context: 202,752 tokens (full)
- Temperature: 0.7 (tool-calling optimized)
- Top-P: 1.0
- Host: 0.0.0.0 (network accessible)

## Storage

Models are stored in `./models/` on the host and mounted into containers:

```
./models/
├── GLM-4.7-Flash-UD-Q4_K_XL.gguf  (~18GB)
└── GLM-4.7-Flash-UD-Q8_K_XL.gguf  (~35GB)
```

This allows model sharing between containers and persists across container restarts.
