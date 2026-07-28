# SUPIR (CVPR 2024) diffusion restoration — SDXL-based, ~21.5 GB of checkpoints.
#
# Same Blackwell-proven base as ../Dockerfile and open-genmoji: CUDA 12.8 +
# torch 2.7 cu128 wheels carry SM_120 kernels for the RTX PRO 6000.
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
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 12.8 wheels — has Blackwell SM_120 kernels.
RUN pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.*" "torchvision==0.22.*"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Fail the BUILD if anything above dragged torch off the cu128 SM_120 wheel.
# Without this, a silent downgrade only surfaces later as
# "CUDA error: no kernel image is available for execution on the device".
RUN python3 -c "\
import torch, sys; v = torch.__version__; \
print('torch', v); \
sys.exit('FATAL: torch was downgraded to ' + v + ' (need 2.7.x+cu128)') \
    if not (v.startswith('2.7') and 'cu128' in v) else None"

# SUPIR source, pinned. Upstream is dormant (last commit 2025-05-12) so this
# is a stable pin rather than a moving target.
ARG SUPIR_SHA=bda91af2000042f8bedfec8897d92917e67c1d88
RUN git clone https://github.com/Fanghua-Yu/SUPIR.git /app/supir_src \
    && cd /app/supir_src \
    && git checkout -q ${SUPIR_SHA} \
    && rm -rf .git

# Two upstream bugs make the tiled-VAE path unusable without xformers; see the
# module docstring for the full diagnosis. Self-verifying: aborts the build if
# upstream's code no longer matches what it expects to patch.
COPY patch_tilevae.py .
RUN python3 patch_tilevae.py

# sgm/modules/encoders/modules.py does `from CKPT_PTH import ...` at import time,
# so this module must exist on sys.path with paths that resolve in-container.
# LLaVA paths are left as None — the captioner is never loaded (prompts come
# over the API instead), and nothing dereferences them on that path.
RUN printf '%s\n' \
    "LLAVA_CLIP_PATH = None" \
    "LLAVA_MODEL_PATH = None" \
    "SDXL_CLIP1_PATH = '/weights/clip-vit-large-patch14'" \
    "SDXL_CLIP2_CKPT_PTH = '/weights/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'" \
    > /app/supir_src/CKPT_PTH.py

ENV PYTHONPATH=/app/supir_src

COPY app ./app

RUN mkdir -p /weights

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
