# Dockerfile for GLM-4.7-Flash Server
# Based on NVIDIA CUDA runtime

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    libcurl4-openssl-dev \
    pciutils \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone and build llama.cpp with Blackwell compatibility
RUN git clone https://github.com/ggml-org/llama.cpp && \
    cmake llama.cpp -B llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" && \
    cmake --build llama.cpp/build --config Release -j$(nproc)

# Install HuggingFace CLI and jq for scripts
RUN pip3 install --no-cache-dir huggingface-hub && \
    apt-get update && apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

# Copy scripts
COPY download_model.sh /app/
COPY start_server.sh /app/
COPY start_server_q8.sh /app/
COPY test_server.sh /app/

# Make scripts executable
RUN chmod +x /app/download_model.sh /app/start_server.sh /app/start_server_q8.sh /app/test_server.sh

# Create directory for models
RUN mkdir -p /app/models

# Expose server port
EXPOSE 11346

# Default command shows usage
CMD ["/bin/bash", "-c", "echo 'GLM-4.7-Flash Docker Container' && echo '' && echo 'Usage with docker-compose:' && echo '  Q4 (4-bit): docker compose up glm-q4' && echo '  Q8 (8-bit): docker compose up glm-q8' && echo '  Download models: docker compose run --rm glm-q4 ./download_model.sh' && echo '' && echo 'Models stored in ./models/ directory'"]
