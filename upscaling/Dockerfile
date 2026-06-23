# Same Blackwell-proven base as open-genmoji: CUDA 12.8 + torch 2.7 cu128 wheels
# carry SM_120 kernels for the RTX PRO 6000.
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
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 12.8 wheels — has Blackwell SM_120 kernels.
RUN pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.*" "torchvision==0.22.*"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

RUN mkdir -p /weights

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
