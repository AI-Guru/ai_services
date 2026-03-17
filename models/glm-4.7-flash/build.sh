#!/bin/bash

# Build llama.cpp with CUDA support and Blackwell compatibility
# This uses explicit CUDA architectures to avoid compute_120a issues

set -e

echo "Building llama.cpp with CUDA support..."
echo "Using CUDA architectures: 80, 86, 89, 90 (Blackwell compatibility mode)"

cmake llama.cpp -B llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"

echo "Starting build process (this may take several minutes)..."
cmake --build llama.cpp/build --config Release -j $(nproc)

echo ""
echo "Build complete!"
echo "llama-server location: llama.cpp/build/bin/llama-server"
