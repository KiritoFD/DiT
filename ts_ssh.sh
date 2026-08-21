#!/bin/sh
echo "=== ensure up --ssh ==="
tailscale --socket=/var/run/tailscale/tailscaled.sock up --ssh --timeout 15s 2>&1 | head
sleep 2
echo "=== status w/ ssh ==="
tailscale --socket=/var/run/tailscale/tailscaled.sock status --json 2>/dev/null | tr ',' '\n' | grep -iE '"SSHEnabled"|"Online"|"CurrentTailnet"|"TailscaleIPs"' | head
echo "=== END ==="
