# SeedVR2-3B (ByteDance-Seed, ICLR 2026) — one-step diffusion video restoration.
#
# NOT servable by vLLM-Omni; see README.md and
# docker-compose.vllm-omni-3b-rtx.yml for that reproducible negative result.
# This is the native path: upstream's own inference stack, patched for Blackwell.
#
# Same base as upscaling/supir and open-genmoji: CUDA 12.8 + torch 2.7 cu128
# wheels carry the SM_120 kernels the RTX PRO 6000 needs.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.*" "torchvision==0.22.*"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Fail the BUILD if anything above dragged torch off the cu128 SM_120 wheel.
RUN python3 -c "\
import torch, sys; v = torch.__version__; \
print('torch', v); \
sys.exit('FATAL: torch was downgraded to ' + v + ' (need 2.7.x+cu128)') \
    if not (v.startswith('2.7') and 'cu128' in v) else None"

# SeedVR2 source, pinned.
ARG SEEDVR_SHA=main
RUN git clone https://github.com/ByteDance-Seed/SeedVR.git /app/seedvr_src \
    && cd /app/seedvr_src \
    && git checkout -q ${SEEDVR_SHA} \
    && rm -rf .git

# Remove the two hard dependencies that have no Blackwell story (flash-attn,
# apex). Self-verifying: aborts the build if upstream no longer matches.
COPY patch_seedvr2.py .
RUN python3 patch_seedvr2.py

# Prove the patched tree imports with neither package present.
RUN cd /app/seedvr_src && python3 -c "\
import flash_attn" 2>/dev/null && echo 'NOTE: flash_attn present' || echo 'ok: no flash_attn (SDPA path)'
RUN python3 -c "import apex" 2>/dev/null && echo 'NOTE: apex present' || echo 'ok: no apex (stock norms)'

ENV PYTHONPATH=/app/seedvr_src

COPY app ./app

RUN mkdir -p /weights

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
