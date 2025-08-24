#!/usr/bin/env bash
set -Eeuo pipefail

echo "⚠️  DEPRECATED: This script has been replaced with a new dev/prod setup!"
echo ""
echo "📋 New Scripts Available:"
echo "  ./start_vllm_tmux.sh  - Start vLLM (one-time setup)"
echo "  ./run_dev.sh          - Development environment (hot-reload)"
echo "  ./run_prod.sh         - Production environment (tmux)"
echo "  ./setup_help.sh       - Show detailed help and status"
echo ""
echo "💡 Quick Start:"
echo "  1. ./start_vllm_tmux.sh  (first time only)"
echo "  2. ./run_dev.sh          (for development)"
echo ""
echo "Would you like to:"
echo "  [h] Show detailed help"
echo "  [d] Start development environment"
echo "  [p] Start production environment"
echo "  [v] Start vLLM only"
echo "  [q] Quit"
echo ""
read -n 1 -r choice
echo

case "$choice" in
    h|H)
        ./setup_help.sh
        ;;
    d|D)
        echo "🛠️  Starting development environment..."
        ./run_dev.sh
        ;;
    p|P)
        echo "🏭 Starting production environment..."
        ./run_prod.sh
        ;;
    v|V)
        echo "🧠 Starting vLLM..."
        ./start_vllm_tmux.sh
        ;;
    *)
        echo "👋 Exiting..."
        exit 0
        ;;
esac
