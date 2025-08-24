#!/usr/bin/env bash

echo "🔍 Anton Network Connectivity Test"
echo "=================================="
echo ""

# Get the current IP address
CURRENT_IP=$(ip route get 1 | awk '{print $7; exit}')
echo "🖥️  Detected IP Address: $CURRENT_IP"
echo ""

# Test localhost connectivity
echo "📡 Testing localhost connectivity..."
if curl -fsS "http://localhost:8001/health" >/dev/null 2>&1; then
    echo "✅ localhost:8001 (dev agent) - ACCESSIBLE"
else
    echo "❌ localhost:8001 (dev agent) - NOT ACCESSIBLE"
fi

if curl -fsS "http://localhost:7860" >/dev/null 2>&1; then
    echo "✅ localhost:7860 (dev UI) - ACCESSIBLE"
else
    echo "❌ localhost:7860 (dev UI) - NOT ACCESSIBLE"
fi

if curl -fsS "http://localhost:9001/health" >/dev/null 2>&1; then
    echo "✅ localhost:9001 (prod agent) - ACCESSIBLE"
else
    echo "❌ localhost:9001 (prod agent) - NOT ACCESSIBLE"
fi

if curl -fsS "http://localhost:9860" >/dev/null 2>&1; then
    echo "✅ localhost:9860 (prod UI) - ACCESSIBLE"
else
    echo "❌ localhost:9860 (prod UI) - NOT ACCESSIBLE"
fi

echo ""
echo "📡 Testing network interface connectivity..."

# Test network interface connectivity (192.168.1.250)
if curl -fsS "http://192.168.1.250:8001/health" >/dev/null 2>&1; then
    echo "✅ 192.168.1.250:8001 (dev agent) - ACCESSIBLE"
else
    echo "❌ 192.168.1.250:8001 (dev agent) - NOT ACCESSIBLE"
fi

if curl -fsS "http://192.168.1.250:7860" >/dev/null 2>&1; then
    echo "✅ 192.168.1.250:7860 (dev UI) - ACCESSIBLE"
else
    echo "❌ 192.168.1.250:7860 (dev UI) - NOT ACCESSIBLE"
fi

if curl -fsS "http://192.168.1.250:9001/health" >/dev/null 2>&1; then
    echo "✅ 192.168.1.250:9001 (prod agent) - ACCESSIBLE"
else
    echo "❌ 192.168.1.250:9001 (prod agent) - NOT ACCESSIBLE"
fi

if curl -fsS "http://192.168.1.250:9860" >/dev/null 2>&1; then
    echo "✅ 192.168.1.250:9860 (prod UI) - ACCESSIBLE"
else
    echo "❌ 192.168.1.250:9860 (prod UI) - NOT ACCESSIBLE"
fi

echo ""
echo "🔧 Checking port bindings..."
echo "Active services on ports 8001, 7860, 9001, 9860:"
netstat -tlnp 2>/dev/null | grep -E ':(8001|7860|9001|9860)\s' || echo "No services found on these ports"

echo ""
echo "🛠️  Troubleshooting Tips:"
echo "========================"
echo ""
echo "If services are accessible on localhost but not on 192.168.1.250:"
echo "1. Check firewall: sudo ufw status"
echo "2. Check if services are binding to 0.0.0.0 (not just 127.0.0.1)"
echo "3. Restart production with updated script: ./run_prod.sh"
echo ""
echo "If no services are running:"
echo "1. Start development: ./run_dev.sh"
echo "2. Start production: ./run_prod.sh"
echo "3. Check tmux sessions: tmux list-sessions"
echo ""
