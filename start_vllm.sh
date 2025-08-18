#!/bin/bash

# vLLM Startup Script for Anton AI Assistant
# This script starts the vLLM server with optimized settings

set -e

# Configuration
VLLM_HOST=${VLLM_HOST:-"0.0.0.0"}
VLLM_PORT=${VLLM_PORT:-8003}
VLLM_MODEL=${VLLM_MODEL:-"Qwen/Qwen2.5-Coder-32B-Instruct"}
VLLM_API_KEY=${VLLM_API_KEY:-"anton-vllm-key"}

# Performance settings
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

echo "🚀 Starting vLLM server for Anton AI Assistant..."
echo "Model: $VLLM_MODEL"
echo "Host: $VLLM_HOST:$VLLM_PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Max Context Length: $MAX_MODEL_LEN"

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "❌ vLLM not found. Installing..."
    pip install vllm[openai]
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🔧 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
fi

# Start vLLM server
exec vllm serve "$VLLM_MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-prefix-caching \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --api-key "$VLLM_API_KEY" \
    --served-model-name "qwen-coder-32b" \
    --trust-remote-code \
    --disable-log-stats
