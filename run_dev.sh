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

# ── Dev Environment Variables ────────────────────────────────────────────────
export ANTON_ENV="dev"
export VLLM_PORT=8003
export AGENT_PORT=8001
export UI_PORT=7860

# ── Env ────────────────────────────────────────────────────────────────────────
if [[ -f .env.vllm ]]; then
  echo "🔧 Loading vLLM environment from .env.vllm..."
  set -o allexport; source .env.vllm; set +o allexport
elif [[ -f .env ]]; then
  echo "🔧 Loading default environment from .env..."
  set -o allexport; source <(grep -v '^\s*#' .env | grep -v '^\s*$'); set +o allexport
fi

LOGDIR=logs
mkdir -p "$LOGDIR"

VLLM_HOSTNAME="${VLLM_HOSTNAME:-localhost}"
HEALTH_CHECK_URL="http://${VLLM_HOSTNAME}:${VLLM_PORT}"

# ── Commands ──────────────────────────────────────────────────────────────────
CMD_AGENT=(uvicorn server.agent.agent_server:app --host 0.0.0.0 --port $AGENT_PORT)
CMD_UI=(chainlit run app.py --headless --port $UI_PORT --host 0.0.0.0)

pid_agent=""
pid_ui=""

# ── Helpers ───────────────────────────────────────────────────────────────────
is_up() { curl -fsS "$1" >/dev/null 2>&1; }
is_proc_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" >/dev/null 2>&1; }

start_agent() {
  echo "▶️  Starting Agent (dev)..."
  export AGENT_HOST=localhost
  "${CMD_AGENT[@]}" >"$LOGDIR/agent_dev.log" 2>&1 & pid_agent=$!
  echo "    Agent PID: $pid_agent (logs: $LOGDIR/agent_dev.log)"
}

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

start_ui() {
  echo "▶️  Starting Chainlit UI (dev)..."
  "${CMD_UI[@]}" >"$LOGDIR/ui_dev.log" 2>&1 & pid_ui=$!
  echo "    UI PID: $pid_ui (logs: $LOGDIR/ui_dev.log)"
}

stop_agent() {
  [[ -n "$pid_agent" ]] && is_proc_alive "$pid_agent" && { kill "$pid_agent" || true; wait "$pid_agent" 2>/dev/null || true; }
  pid_agent=""
}
stop_ui() {
  [[ -n "$pid_ui" ]] && is_proc_alive "$pid_ui" && { kill "$pid_ui" || true; wait "$pid_ui" 2>/dev/null || true; }
  pid_ui=""
}
restart_agent_only() { stop_agent; start_agent; }
restart_ui_only()    { stop_ui;    start_ui; }
restart_agent_ui()   { stop_agent; stop_ui; start_agent; start_ui; }

cleanup() {
  echo -e "\n🛑 Stopping dev services…"
  stop_agent
  stop_ui
}
trap cleanup INT TERM EXIT

# ── Pre-reqs ──────────────────────────────────────────────────────────────────
docker start chainlit-pg >/dev/null 2>&1 || true
sudo chmod -R 777 /home/lucas/anton_new/pgdata || true

# ── Boot ──────────────────────────────────────────────────────────────────────
echo "🚀 Starting Anton AI Assistant (DEV) services..."
check_vllm
start_agent
start_ui

echo ""
echo "✅ All DEV services started!"
echo "   🌐 Web UI:  http://localhost:${UI_PORT}"
echo "   🤖 Agent:   http://localhost:${AGENT_PORT}"
echo "   🧠 vLLM:    ${HEALTH_CHECK_URL}"
echo ""
echo "Controls: [r] restart Agent+UI   [a] Agent only   [u] UI only   [q] quit"
echo ""

# ── Interactive hotkeys loop + light liveness polling ────────────────────────
while true; do
  # Non-blocking read with a small timeout so we can also poll liveness
  if IFS= read -rsn1 -t 0.3 key; then
    case "$key" in
      r) echo "🔁 Restarting Agent+UI…"; restart_agent_ui ;;
      a) echo "🔁 Restarting Agent…";    restart_agent_only ;;
      u) echo "🔁 Restarting UI…";       restart_ui_only ;;
      q) echo "👋 Quitting…";            break ;;
      *) : ;;
    esac
  fi

  # If any managed child dies, show a note (don't exit; you can hot-restart)
  if [[ -n "$pid_agent" ]] && ! is_proc_alive "$pid_agent"; then
    echo "⚠️  Agent exited. Press 'a' to restart, or 'r' for Agent+UI."
    pid_agent=""
  fi
  if [[ -n "$pid_ui" ]] && ! is_proc_alive "$pid_ui"; then
    echo "⚠️  UI exited. Press 'u' to restart, or 'r' for Agent+UI."
    pid_ui=""
  fi
done
