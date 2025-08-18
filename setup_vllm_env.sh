#!/bin/bash

# Environment Setup for vLLM Migration
# Sets up all necessary environment variables for Anton to use vLLM

# vLLM Server Configuration
export VLLM_HOST="http://localhost:8003"
export VLLM_API_KEY="anton-vllm-key" 
export VLLM_MODEL="qwen-coder-32b"

# Performance Tuning
export VLLM_MAX_MODEL_LEN="32768"
export VLLM_TENSOR_PARALLEL_SIZE="1"
export VLLM_GPU_MEMORY_UTILIZATION="0.8"

# Anton Configuration  
export ANTON_MD_DEBUG="0"
export LOG_LEVEL="INFO"

echo "✅ Environment configured for vLLM backend"
echo "   VLLM_HOST: $VLLM_HOST"
echo "   VLLM_MODEL: $VLLM_MODEL"
echo "   Max Context: $VLLM_MAX_MODEL_LEN tokens"

# Save to .env file for persistence
cat > .env.vllm << EOF
# vLLM Configuration for Anton AI Assistant
VLLM_HOST=http://localhost:8003
VLLM_API_KEY=anton-vllm-key
VLLM_MODEL=qwen-coder-32b
VLLM_MAX_MODEL_LEN=32768
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.8
ANTON_MD_DEBUG=0
LOG_LEVEL=INFO
EOF

echo "   Configuration saved to .env.vllm"
echo ""
echo "🔄 To apply these settings, run:"
echo "   source setup_vllm_env.sh"
echo ""
echo "🚀 To start vLLM server, run:"  
echo "   ./start_vllm.sh"
