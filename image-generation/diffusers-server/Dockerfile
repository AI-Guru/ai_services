# Reusable OpenAI-compatible diffusers image server (text-to-image).
#
# Base: the vLLM cu130 image, which already ships a Blackwell / SM_120-capable
# PyTorch (torch 2.11+cu130). Verified to load and generate Z-Image-Turbo on the
# RTX PRO 6000 (sm_120) — see models/z-image/out/smoke.log. Reusing it means we
# don't have to re-prove the CUDA stack.
FROM vllm/vllm-openai:cu130-nightly

# diffusers from source: ZImagePipeline (and other 2026 pipelines) ship ahead of
# the last PyPI release. --no-deps so we never disturb the working SM_120 torch
# stack; diffusers' runtime deps (transformers, accelerate, safetensors, etc.)
# are already present in the base image.
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --no-deps "git+https://github.com/huggingface/diffusers" \
    && pip install --no-cache-dir fastapi "uvicorn[standard]"

COPY server.py /app/server.py
WORKDIR /app

# Override the base image's vLLM entrypoint with our server.
ENTRYPOINT []
CMD ["python3", "server.py"]
