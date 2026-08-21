#!/bin/sh
pkill -f tailscaled 2>/dev/null
sleep 1
mkdir -p /var/lib/tailscale /var/run/tailscale
/usr/sbin/tailscaled \
  --state=/var/lib/tailscale/tailscaled.state \
  --socket=/var/run/tailscale/tailscaled.sock \
  --tun=userspace-networking \
  --port=0 \
  > /var/log/tailscaled3.log 2>&1 &
TPID=$!
sleep 4
echo "=== tailscaled PID=$TPID ==="
kill -0 $TPID 2>/dev/null && echo "ALIVE" || echo "DEAD"
echo "=== tailscale up (nodes mode) ==="
timeout 10 tailscale --socket=/var/run/tailscale/tailscaled.sock up --ssh --timeout 8s 2>&1 | head -n30
echo "=== status ==="
tailscale --socket=/var/run/tailscale/tailscaled.sock status 2>&1 | head -n5
echo "=== END ==="
