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

# ── Env ────────────────────────────────────────────────────────────────────────
if [[ -f .env.vllm ]]; then
  echo "🔧 Loading vLLM environment from .env.vllm..."
  set -o allexport; source .env.vllm; set +o allexport
elif [[ -f .env ]]; then
  echo "🔧 Loading default environment from .env..."
  set -o allexport; source <(grep -v '^\s*#' .env | grep -v '^\s*$'); set +o allexport
fi

VLLM_HOSTNAME="${VLLM_HOSTNAME:-localhost}"
VLLM_PORT="${VLLM_PORT:-8003}"
TMUX_SESSION_NAME="anton-vllm"

# Check if tmux session already exists
if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
    echo "⚠️  vLLM tmux session '$TMUX_SESSION_NAME' already exists!"
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

# Pre-requisites
docker start chainlit-pg >/dev/null 2>&1 || true
sudo chmod -R 777 /home/lucas/anton_new/pgdata || true

# Start vLLM in a new tmux session
echo "🚀 Starting vLLM in tmux session '$TMUX_SESSION_NAME'..."
echo "   🧠 vLLM will be available at: http://${VLLM_HOSTNAME}:${VLLM_PORT}"
echo ""
echo "📝 To monitor vLLM:"
echo "   tmux attach-session -t $TMUX_SESSION_NAME"
echo ""
echo "📝 To stop vLLM:"
echo "   tmux kill-session -t $TMUX_SESSION_NAME"
echo ""

# Create tmux session and start vLLM
tmux new-session -d -s "$TMUX_SESSION_NAME" -c "$(pwd)"
tmux send-keys -t "$TMUX_SESSION_NAME" "source .venv/bin/activate" Enter

# Load environment in tmux session
if [[ -f .env.vllm ]]; then
    tmux send-keys -t "$TMUX_SESSION_NAME" "set -o allexport; source .env.vllm; set +o allexport" Enter
elif [[ -f .env ]]; then
    tmux send-keys -t "$TMUX_SESSION_NAME" "set -o allexport; source <(grep -v '^\s*#' .env | grep -v '^\s*$'); set +o allexport" Enter
fi

# Start vLLM
tmux send-keys -t "$TMUX_SESSION_NAME" "./start_vllm.sh" Enter

echo "✅ vLLM started in background tmux session!"
echo "💡 You can now run './run_dev.sh' or './run_prod.sh' to start the application."
