#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"   # project root (put this file there)

# Check if vLLM environment is configured
if [[ -f .env.vllm ]]; then
    echo "🔧 Loading vLLM environment..."
    set -o allexport
    source .env.vllm
    set +o allexport
elif [[ -f .env ]]; then
    echo "🔧 Loading default environment..."
    set -o allexport
    source <(grep -v '^\s*#' .env | grep -v '^\s*$')
    set +o allexport
fi

# Check if vLLM server is running
if [[ -n "${VLLM_HOST:-}" ]]; then
    echo "🔍 Checking vLLM server at ${VLLM_HOST}..."
    if ! curl -s "${VLLM_HOST}/health" > /dev/null 2>&1; then
        echo "❌ vLLM server not responding at ${VLLM_HOST}"
        echo "   Start vLLM server first with: ./start_vllm.sh"
        echo "   Or set up environment with: ./setup_vllm_env.sh"
        exit 1
    else
        echo "✅ vLLM server is running"
    fi
fi

LOGDIR=logs
mkdir -p "$LOGDIR"

echo "🚀 Starting Anton AI Assistant services..."

cmd1="uvicorn server.agent.agent_server:app --host 0.0.0.0 --port 8001 --reload"
cmd2="python3 -m server.model_server"
cmd3="chainlit run app.py --port 7860 --host 0.0.0.0"

pids=()
cleanup() {
  echo "🛑 Stopping services..."
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup INT TERM EXIT

echo "📋 Starting services..."
$cmd1 >"$LOGDIR/agent.log"  2>&1 & pids+=($!)
echo "   Agent Server: PID ${pids[0]} (logs: $LOGDIR/agent.log)"

$cmd2 >"$LOGDIR/model.log"  2>&1 & pids+=($!)
echo "   Model Server: PID ${pids[1]} (logs: $LOGDIR/model.log)"

$cmd3 >"$LOGDIR/ui.log"     2>&1 & pids+=($!)
echo "   Chainlit UI:  PID ${pids[2]} (logs: $LOGDIR/ui.log)"

echo ""
echo "✅ All services started successfully!"
echo "   🌐 Web UI: http://localhost:7860"
echo "   🤖 Agent API: http://localhost:8001"
echo "   🧠 Model API: http://localhost:8002"
echo ""
echo "📊 To monitor performance:"
echo "   python benchmark_vllm.py"
echo ""
echo "🧪 To run migration tests:"
echo "   python test_vllm_migration.py"
echo ""
echo "Press Ctrl+C to stop all services..."

wait -n  # if any dies, we kill the rest via trap
