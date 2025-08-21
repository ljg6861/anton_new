#!/bin/bash

# vLLM Startup Script for Anton AI Assistant
set -e

# --- Sanity Check: Ensure we're in a virtual environment ---
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ This script should be run inside a Python virtual environment."
    echo "💡 Please activate your venv first."
    exit 1
fi

# Configuration from environment variables
VLLM_HOSTNAME=${VLLM_HOSTNAME:-"0.0.0.0"}
VLLM_PORT=${VLLM_PORT:-8003}
VLLM_MODEL=${VLLM_MODEL:-"models/qwen3-coder-30b-a3b-instruct-q4_k_m.gguf"}
VLLM_SERVED_MODEL_NAME="anton"
VLLM_API_KEY=${VLLM_API_KEY:-"anton-vllm-key"}

# Performance settings from environment variables
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20976}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}

# --- GPU Configuration ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# If using more than one GPU, make them all visible to CUDA
if [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    # Create a comma-separated list of GPU IDs from 0 to TENSOR_PARALLEL_SIZE-1
    # Example: if TENSOR_PARALLEL_SIZE=2, this creates "0,1"
    VISIBLE_DEVICES=$(seq -s, 0 $((TENSOR_PARALLEL_SIZE - 1)))
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$VISIBLE_DEVICES}
    echo "🔧 Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

echo "🚀 Starting vLLM server for Anton AI Assistant..."
echo "Model: $VLLM_MODEL"
echo "Host: $VLLM_HOSTNAME:$VLLM_PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Max Context Length: $MAX_MODEL_LEN"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🔧 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
fi

# Start vLLM server
exec vllm serve "$VLLM_MODEL" \
    --host "$VLLM_HOSTNAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --api-key "$VLLM_API_KEY" \
    --served-model-name "$VLLM_SERVED_MODEL_NAME" \
    --trust-remote-code \
    --disable-log-stats \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser "qwen3_coder"