#!/bin/sh
echo "=== attempt foreground tailscaled ==="
pkill -f tailscaled 2>/dev/null
sleep 1
mkdir -p /var/lib/tailscale /var/run/tailscale
/usr/sbin/tailscaled --state=/var/lib/tailscale/tailscaled.state --socket=/var/run/tailscale/tailscaled.sock --port=0 > /var/log/tailscaled2.log 2>&1 &
TPID=$!
sleep 4
echo "=== tailscaled PID=$TPID alive? ==="
if kill -0 $TPID 2>/dev/null; then echo "ALIVE"; else echo "DEAD exit=$?"; fi
echo "=== log ==="
cat /var/log/tailscaled2.log 2>&1 | head -n20
echo "=== try tailscale up for the URL ==="
timeout 5 tailscale --socket=/var/run/tailscale/tailscaled.sock up --qr 2>&1 | head -n20
echo "=== END ==="
