#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"  # project root

# ── Activate virtual environment ─────────────────────────────────────────────
if [[ -d ".venv" ]]; then
  echo "🐍 Activating virtual environment..."
  source .venv/bin/activate
else
  echo "⚠️  No virtual environment found (.venv directory missing)"
  exit 1
fi

# ── Prod Environment Variables ────────────────────────────────────────────────
export ANTON_ENV="prod"
export VLLM_PORT=8003  # vLLM remains on same port (shared)
export AGENT_PORT=9001  # Different port for prod
export UI_PORT=9860     # Different port for prod

# ── Env ────────────────────────────────────────────────────────────────────────
if [[ -f .env.vllm ]]; then
  echo "🔧 Loading vLLM environment from .env.vllm..."
  set -o allexport; source .env.vllm; set +o allexport
elif [[ -f .env ]]; then
  echo "🔧 Loading default environment from .env..."
  set -o allexport; source <(grep -v '^\s*#' .env | grep -v '^\s*$'); set +o allexport
fi

VLLM_HOSTNAME="${VLLM_HOSTNAME:-localhost}"
HEALTH_CHECK_URL="http://${VLLM_HOSTNAME}:${VLLM_PORT}"
TMUX_SESSION_NAME="anton-prod"

# Check if tmux session already exists
if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Production tmux session '$TMUX_SESSION_NAME' already exists!"
    echo "Would you like to:"
    echo "  [k] Kill existing session and start fresh"
    echo "  [a] Attach to existing session"
    echo "  [q] Quit"
    read -n 1 -r choice
    echo
    case "$choice" in
        k|K)
            echo "🔪 Killing existing session..."
            tmux kill-session -t "$TMUX_SESSION_NAME"
            ;;
        a|A)
            echo "🔗 Attaching to existing session..."
            tmux attach-session -t "$TMUX_SESSION_NAME"
            exit 0
            ;;
        *)
            echo "👋 Exiting..."
            exit 0
            ;;
    esac
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
is_up() { curl -fsS "$1" >/dev/null 2>&1; }

check_vllm() {
  echo "🔍 Checking vLLM at ${HEALTH_CHECK_URL}/health..."
  if ! is_up "${HEALTH_CHECK_URL}/health"; then
    echo "❌ vLLM not responding at ${HEALTH_CHECK_URL}"
    echo "💡 Please start vLLM first with: ./start_vllm_tmux.sh"
    exit 1
  else
    echo "✅ vLLM is running."
  fi
}

# ── Pre-reqs ──────────────────────────────────────────────────────────────────
docker start chainlit-pg >/dev/null 2>&1 || true
sudo chmod -R 777 /home/lucas/anton_new/pgdata || true

# ── Boot ──────────────────────────────────────────────────────────────────────
echo "🚀 Starting Anton AI Assistant (PROD) services..."
check_vllm

# Create tmux session with multiple windows
echo "📺 Creating tmux session '$TMUX_SESSION_NAME' with Agent and UI..."
tmux new-session -d -s "$TMUX_SESSION_NAME" -c "$(pwd)"

# Setup first window for Agent
tmux rename-window -t "$TMUX_SESSION_NAME:0" "agent"
tmux send-keys -t "$TMUX_SESSION_NAME:agent" "source .venv/bin/activate" Enter

# Load environment in agent window
if [[ -f .env.vllm ]]; then
    tmux send-keys -t "$TMUX_SESSION_NAME:agent" "set -o allexport; source .env.vllm; set +o allexport" Enter
elif [[ -f .env ]]; then
    tmux send-keys -t "$TMUX_SESSION_NAME:agent" "set -o allexport; source <(grep -v '^\s*#' .env | grep -v '^\s*$'); set +o allexport" Enter
fi

# Set environment variables and start agent
tmux send-keys -t "$TMUX_SESSION_NAME:agent" "export ANTON_ENV=prod AGENT_PORT=$AGENT_PORT VLLM_PORT=$VLLM_PORT UI_PORT=$UI_PORT AGENT_HOST=192.168.1.250" Enter
tmux send-keys -t "$TMUX_SESSION_NAME:agent" "uvicorn server.agent.agent_server:app --host 0.0.0.0 --port $AGENT_PORT" Enter

# Create second window for UI
tmux new-window -t "$TMUX_SESSION_NAME" -n "ui" -c "$(pwd)"
tmux send-keys -t "$TMUX_SESSION_NAME:ui" "source .venv/bin/activate" Enter

# Load environment in UI window
if [[ -f .env.vllm ]]; then
    tmux send-keys -t "$TMUX_SESSION_NAME:ui" "set -o allexport; source .env.vllm; set +o allexport" Enter
elif [[ -f .env ]]; then
    tmux send-keys -t "$TMUX_SESSION_NAME:ui" "set -o allexport; source <(grep -v '^\s*#' .env | grep -v '^\s*$'); set +o allexport" Enter
fi

# Set environment variables and start UI
tmux send-keys -t "$TMUX_SESSION_NAME:ui" "export ANTON_ENV=prod AGENT_PORT=$AGENT_PORT VLLM_PORT=$VLLM_PORT UI_PORT=$UI_PORT AGENT_HOST=192.168.1.250" Enter
tmux send-keys -t "$TMUX_SESSION_NAME:ui" "chainlit run app.py --headless --port $UI_PORT --host 0.0.0.0" Enter

# Wait a moment for services to start
sleep 5

echo ""
echo "✅ All PROD services started in tmux!"
echo "   🌐 Web UI:  http://192.168.1.250:${UI_PORT}"
echo "   🤖 Agent:   http://192.168.1.250:${AGENT_PORT}" 
echo "   🧠 vLLM:    ${HEALTH_CHECK_URL}"
echo ""
echo "🔍 Verifying services are accessible..."

# Check if services are responding on the network interface
sleep 3
if curl -fsS "http://192.168.1.250:${AGENT_PORT}/health" >/dev/null 2>&1; then
    echo "✅ Agent accessible on network interface"
else
    echo "⚠️  Agent not responding on network interface - check firewall/network settings"
fi

if curl -fsS "http://192.168.1.250:${UI_PORT}" >/dev/null 2>&1; then
    echo "✅ UI accessible on network interface"
else
    echo "⚠️  UI not responding on network interface - check firewall/network settings"
fi
echo ""
echo "📝 To monitor production:"
echo "   tmux attach-session -t $TMUX_SESSION_NAME"
echo ""
echo "📝 To stop production:"
echo "   tmux kill-session -t $TMUX_SESSION_NAME"
echo ""
echo "🎛️  Tmux Controls (when attached):"
echo "   Ctrl+b, w   - List and switch between windows"
echo "   Ctrl+b, 0   - Switch to agent window"
echo "   Ctrl+b, 1   - Switch to UI window"
echo "   Ctrl+b, d   - Detach from session (leave running)"
echo ""

# Optionally attach to the session
echo "Would you like to attach to the session now? [y/N]"
read -n 1 -r attach_choice
echo
if [[ "$attach_choice" =~ ^[Yy]$ ]]; then
    echo "🔗 Attaching to production session..."
    tmux attach-session -t "$TMUX_SESSION_NAME"
else
    echo "👍 Production is running in background. Use 'tmux attach-session -t $TMUX_SESSION_NAME' to monitor."
fi
