#!/bin/sh
echo "PID1=$(tr -d '\0' < /proc/1/comm)"
systemctl is-system-running 2>&1 || true
# Try to start tailscaled via systemd if available, else manually
if command -v systemctl >/dev/null 2>&1 && systemctl start tailscaled 2>/dev/null; then
  echo "tailscaled started via systemd"
else
  echo "systemd start failed or unavailable; starting tailscaled manually"
  pkill -f tailscaled 2>/dev/null; sleep 1
  nohup /usr/sbin/tailscaled > /var/log/tailscaled.log 2>&1 &
  sleep 3
fi
tailscale version 2>&1
echo "--- daemon status ---"
tailscale status 2>&1 | head -n3
echo "--- DONE ---"
