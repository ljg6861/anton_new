#!/usr/bin/env bash

# Anton AI Assistant - Setup Helper
# This script helps you understand and manage the new dev/prod environment setup

echo "🚀 Anton AI Assistant - Environment Setup"
echo "=========================================="
echo ""
echo "📋 Available Scripts:"
echo ""
echo "1. ./start_vllm_tmux.sh"
echo "   🧠 Starts vLLM (AI model) in a persistent tmux session"
echo "   ⚡ This only needs to be run ONCE and will persist across reboots"
echo "   🔄 Shared between dev and prod environments"
echo ""
echo "2. ./run_dev.sh"
echo "   🛠️  Development environment (ports: 8001=agent, 7860=UI, 8003=vLLM)"
echo "   🔥 Hot-reload: press 'r' to restart agent+UI, 'a' for agent only, 'u' for UI only"
echo "   💻 Interactive console - stays open for debugging"
echo ""
echo "3. ./run_prod.sh"
echo "   🏭 Production environment (ports: 9001=agent, 9860=UI, 8003=vLLM)"
echo "   📺 Runs in tmux for persistence - can survive SSH disconnections"
echo "   🔧 Background operation - use tmux commands to monitor"
echo ""
echo "📊 Environment Status:"
echo "====================="
echo ""

# Check if vLLM tmux session exists
if tmux has-session -t "anton-vllm" 2>/dev/null; then
    echo "✅ vLLM: Running in tmux session 'anton-vllm'"
else
    echo "❌ vLLM: Not running (start with ./start_vllm_tmux.sh)"
fi

# Check if prod tmux session exists
if tmux has-session -t "anton-prod" 2>/dev/null; then
    echo "✅ Production: Running in tmux session 'anton-prod'"
    echo "   🌐 Web UI: http://192.168.1.250:9860"
    echo "   🤖 Agent: http://192.168.1.250:9001"
else
    echo "❌ Production: Not running (start with ./run_prod.sh)"
fi

# Check dev ports
if curl -fsS "http://localhost:8001/health" >/dev/null 2>&1; then
    echo "✅ Development Agent: Running on port 8001"
    echo "   🤖 Local: http://localhost:8001"
    echo "   🌐 Network: http://192.168.1.250:8001"
else
    echo "❌ Development Agent: Not running"
fi

if curl -fsS "http://localhost:7860" >/dev/null 2>&1; then
    echo "✅ Development UI: Running on port 7860"
    echo "   🌐 Local: http://localhost:7860"
    echo "   🌐 Network: http://192.168.1.250:7860"
else
    echo "❌ Development UI: Not running"
fi

echo ""
echo "🔧 Common tmux Commands:"
echo "======================="
echo ""
echo "# Attach to vLLM session (monitor model)"
echo "tmux attach-session -t anton-vllm"
echo ""
echo "# Attach to production session (monitor prod)"
echo "tmux attach-session -t anton-prod"
echo ""
echo "# List all tmux sessions"
echo "tmux list-sessions"
echo ""
echo "# Kill a session (stop services)"
echo "tmux kill-session -t anton-vllm    # Stop vLLM"
echo "tmux kill-session -t anton-prod    # Stop production"
echo ""
echo "# Detach from session (Ctrl+b, then d)"
echo "# This leaves the session running in background"
echo ""
echo "💡 Recommended Workflow:"
echo "========================"
echo ""
echo "1. One-time setup: ./start_vllm_tmux.sh"
echo "2. For development: ./run_dev.sh (interactive)"
echo "3. For production: ./run_prod.sh (background)"
echo "4. Use tmux to monitor/manage background services"
echo ""
