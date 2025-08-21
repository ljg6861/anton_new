#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"  # project root


# Load environment file
if [[ -f .env.vllm ]]; then
    echo "🔧 Loading vLLM environment from .env.vllm..."
    set -o allexport
    source .env.vllm
    set +o allexport
elif [[ -f .env ]]; then
    echo "🔧 Loading default environment from .env..."
    set -o allexport
    source <(grep -v '^\s*#' .env | grep -v '^\s*$')
    set +o allexport
fi

# Check if vLLM server is running by constructing the URL
HEALTH_CHECK_URL="http://${VLLM_HOSTNAME:-"localhost"}:${VLLM_PORT:-"8003"}"
echo "🔍 Checking vLLM server at ${HEALTH_CHECK_URL}..."
if ! curl -s "${HEALTH_CHECK_URL}/health" > /dev/null 2>&1; then
    echo "⚠️  vLLM server not responding, will start it automatically..."
else
    echo "✅ vLLM server is already running."
    # If the server is already running, we don't need to do anything else.
    # The original script would start a new one anyway.
    # For a robust script, you might want to exit here or just monitor.
    # For now, we will proceed to launch all services as intended.
fi

LOGDIR=logs
mkdir -p "$LOGDIR"

echo "🚀 Starting Anton AI Assistant services..."
docker start chainlit-pg
sudo chmod -R 777 /home/lucas/anton_new/pgdata
cmd1="uvicorn server.agent.agent_server:app --host 0.0.0.0 --port 8001 --reload"
cmd2="./start_vllm.sh"
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

$cmd2 >"$LOGDIR/vllm.log"  2>&1 & pids+=($!)
echo "   vLLM Server:  PID ${pids[1]} (logs: $LOGDIR/vllm.log)"

$cmd3 >"$LOGDIR/ui.log"    2>&1 & pids+=($!)
echo "   Chainlit UI:  PID ${pids[2]} (logs: $LOGDIR/ui.log)"

echo ""
echo "✅ All services started successfully!"
echo "   🌐 Web UI: http://localhost:7860"
echo "   🤖 Agent API: http://localhost:8001"
echo "   🧠 vLLM API: ${HEALTH_CHECK_URL}"
echo ""
echo "Press Ctrl+C to stop all services..."

wait -n  # if any dies, we kill the rest via trap