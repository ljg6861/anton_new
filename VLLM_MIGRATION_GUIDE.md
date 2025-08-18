# vLLM Migration Guide

Complete guide for migrating Anton AI Assistant from Ollama to vLLM for improved performance and scalability.

## Quick Start

### 1. Install Dependencies
```bash
# Install vLLM with OpenAI compatibility
pip install vllm[openai]

# Ensure Anton requirements are up to date
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Set up environment variables
./setup_vllm_env.sh
source .env.vllm
```

### 3. Start vLLM Server
```bash
# Start vLLM server (will download model if needed)
./start_vllm.sh
```

### 4. Start Anton Services
```bash
# Start Anton with vLLM backend
./run.sh
```

### 5. Verify Migration
```bash
# Run comprehensive tests
python test_vllm_migration.py

# Quick functional test
curl -X POST http://localhost:8001/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello, are you working with vLLM?"}]}'
```

## Architecture Changes

### Before (Ollama)
```
Chainlit → Agent Server → Ollama API
                       ↓
                Tool Executor
```

### After (vLLM)
```
Chainlit → Agent Server → vLLM OpenAI API
                       ↓
                Tool Executor + Native Function Calling
```

## Key Benefits

### Performance Improvements
- **2-3x throughput** improvement via optimized attention and KV caching
- **40-60% latency reduction** through tensor parallelism and efficient memory usage
- **Better GPU utilization** with batched inference and memory management

### Scalability Features
- **Prefix caching** for repeated queries and tool patterns
- **Tensor parallelism** for multi-GPU deployments
- **Dynamic batching** for handling concurrent requests
- **Streaming optimization** for real-time responses

### Enterprise Features
- **OpenAI-compatible API** for easy integration
- **Built-in function calling** with structured tool responses
- **Resource monitoring** and health checks
- **Horizontal scaling** capabilities

## Configuration Options

### Environment Variables

```bash
# vLLM Server Settings
VLLM_HOST=http://localhost:8003          # vLLM server endpoint
VLLM_API_KEY=anton-vllm-key              # API authentication key
VLLM_MODEL=qwen-coder-32b                # Model identifier

# Performance Tuning
VLLM_MAX_MODEL_LEN=32768                 # Maximum context length
VLLM_TENSOR_PARALLEL_SIZE=1              # Number of GPUs for model parallelism
VLLM_GPU_MEMORY_UTILIZATION=0.8          # GPU memory utilization (0.1-0.9)

# Anton Settings
ANTON_MD_DEBUG=0                         # Debug mode (0/1)
LOG_LEVEL=INFO                           # Logging level
```

### Hardware Requirements

#### Minimum (Development)
- **CPU**: 8 cores, 32GB RAM
- **GPU**: RTX 3090 / RTX 4090 (24GB VRAM) for 32B model
- **Storage**: 100GB free space for model downloads

#### Recommended (Production)
- **CPU**: 16+ cores, 64GB+ RAM  
- **GPU**: A100 40GB or H100 for optimal performance
- **Storage**: NVMe SSD with 200GB+ free space
- **Network**: High bandwidth for model serving

#### Multi-GPU Setup
```bash
# For 32B model across 2 GPUs
export VLLM_TENSOR_PARALLEL_SIZE=2

# For 70B model across 4 GPUs  
export VLLM_TENSOR_PARALLEL_SIZE=4
```

## Tool Calling Migration

### Old Format (Ollama Custom)
```python
# Custom parsing of text-based tool calls
if "<tool_call>" in response:
    # Parse custom format
    tool_name, args = parse_custom_format(response)
```

### New Format (OpenAI Standard)
```python
# Native OpenAI function calling
{
  "message": {
    "tool_calls": [
      {
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "list_files",
          "arguments": '{"directory": "."}'
        }
      }
    ]
  }
}
```

### Tool Definition Format
```python
{
  "type": "function",
  "function": {
    "name": "list_files",
    "description": "List files in a directory",
    "parameters": {
      "type": "object",
      "properties": {
        "directory": {
          "type": "string",
          "description": "Directory path to list"
        }
      },
      "required": ["directory"]
    }
  }
}
```

## Performance Tuning

### GPU Memory Optimization
```bash
# Conservative (stable)
export VLLM_GPU_MEMORY_UTILIZATION=0.7

# Aggressive (maximum performance)
export VLLM_GPU_MEMORY_UTILIZATION=0.9
```

### Context Length Tuning
```bash
# Standard context
export VLLM_MAX_MODEL_LEN=16384

# Extended context (requires more memory)
export VLLM_MAX_MODEL_LEN=32768

# Maximum context (memory permitting)
export VLLM_MAX_MODEL_LEN=65536
```

### Caching Configuration
```bash
# Enable prefix caching (recommended)
vllm serve MODEL_NAME --enable-prefix-caching

# Enable chunked prefill for long sequences
vllm serve MODEL_NAME --enable-chunked-prefill

# Speculative decoding (experimental)
vllm serve MODEL_NAME --speculative-model SMALL_MODEL
```

## Monitoring & Observability

### Health Checks
```bash
# vLLM server health
curl http://localhost:8003/health

# Anton agent health  
curl http://localhost:8001/health
```

### Performance Metrics
```bash
# Check vLLM metrics
curl http://localhost:8003/metrics

# GPU utilization
nvidia-smi -l 1

# Memory usage
htop
```

### Logging
```bash
# vLLM server logs
tail -f /var/log/vllm.log

# Anton agent logs
tail -f logs/agent.log

# Model server logs
tail -f logs/model.log
```

## Troubleshooting

### Common Issues

#### "CUDA out of memory"
```bash
# Reduce memory utilization
export VLLM_GPU_MEMORY_UTILIZATION=0.6

# Use smaller context length
export VLLM_MAX_MODEL_LEN=16384

# Enable CPU offloading (if available)
export VLLM_CPU_OFFLOAD=true
```

#### "Model not found"
```bash
# Check model name
vllm serve --help

# List available models
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct --dry-run

# Manually download model
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct
```

#### "Connection refused"
```bash
# Check if vLLM is running
ps aux | grep vllm

# Check port binding
netstat -tulpn | grep 8003

# Restart vLLM server
pkill -f vllm
./start_vllm.sh
```

#### "Tool calls not working"
```bash
# Verify model supports function calling
curl -X POST http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-coder-32b",
    "messages": [{"role": "user", "content": "test"}],
    "tools": [{"type": "function", "function": {"name": "test"}}]
  }'

# Check Anton tool configuration
grep -r "tools" server/agent/
```

### Performance Issues

#### High Latency
```bash
# Enable prefix caching
vllm serve MODEL --enable-prefix-caching

# Increase tensor parallelism
export VLLM_TENSOR_PARALLEL_SIZE=2

# Optimize GPU memory
export VLLM_GPU_MEMORY_UTILIZATION=0.8
```

#### Low Throughput
```bash
# Increase batch size
vllm serve MODEL --max-num-seqs 256

# Enable chunked prefill
vllm serve MODEL --enable-chunked-prefill

# Use multiple GPUs
export VLLM_TENSOR_PARALLEL_SIZE=4
```

## Production Deployment

### Docker Setup
```yaml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8003:8000"
    environment:
      - VLLM_API_KEY=your-secure-key
    command: >
      --model Qwen/Qwen2.5-Coder-32B-Instruct
      --tensor-parallel-size 2
      --enable-prefix-caching
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 2
          requests:
            nvidia.com/gpu: 2
        ports:
        - containerPort: 8000
        command:
          - vllm
          - serve
          - Qwen/Qwen2.5-Coder-32B-Instruct
          - --tensor-parallel-size=2
          - --enable-prefix-caching
```

### Load Balancing
```nginx
upstream vllm_backend {
    server 127.0.0.1:8003;
    server 127.0.0.1:8004;  # Multiple vLLM instances
}

server {
    listen 80;
    location /v1/ {
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Migration Checklist

### Pre-Migration
- [ ] Backup current Ollama configuration
- [ ] Verify hardware requirements
- [ ] Download and test vLLM installation
- [ ] Run baseline performance tests

### Migration Steps
- [ ] Install vLLM and dependencies
- [ ] Update model server configuration
- [ ] Configure environment variables
- [ ] Start vLLM server
- [ ] Update Anton agent code
- [ ] Test tool calling functionality
- [ ] Validate streaming responses

### Post-Migration
- [ ] Run comprehensive test suite
- [ ] Monitor performance metrics
- [ ] Verify all tools working
- [ ] Update documentation
- [ ] Train team on new setup

### Rollback Plan
If issues occur, revert with:
```bash
# Stop vLLM
pkill -f vllm

# Restore Ollama in requirements.txt
pip install ollama

# Revert model_server.py
git checkout HEAD~1 server/model_server.py

# Start Ollama
ollama serve
```

## Support & Resources

### Documentation
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Qwen Model Documentation](https://huggingface.co/Qwen)

### Community
- [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
- [Anton AI Discord](https://discord.gg/anton-ai)

### Professional Support
- Enterprise support available for production deployments
- Custom optimization and scaling consultations
- 24/7 monitoring and incident response

---

*Last updated: August 18, 2025*
*Migration guide version: 1.0*
