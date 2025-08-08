#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"   # project root (put this file there)

LOGDIR=logs
mkdir -p "$LOGDIR"

cmd1="uvicorn server.agent.agent_server:app --host 0.0.0.0 --port 8001 --reload"
cmd2="python3 -m server.model_server"
cmd3="chainlit run app.py --port 7860 --host 0.0.0.0"

pids=()
cleanup() {
  echo "Stopping..."
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup INT TERM EXIT

$cmd1 >"$LOGDIR/agent.log"  2>&1 & pids+=($!)
$cmd2 >"$LOGDIR/model.log"  2>&1 & pids+=($!)
$cmd3 >"$LOGDIR/ui.log"     2>&1 & pids+=($!)

echo "Started: agent=${pids[0]} model=${pids[1]} chainlit=${pids[2]}"
wait -n  # if any dies, we kill the rest via trap
