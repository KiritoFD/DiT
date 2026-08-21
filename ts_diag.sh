#!/bin/sh
S=/var/run/tailscale/tailscaled.sock
echo "== tailscaled log: ssh-related lines =="
LOG=$(ls -t /var/log/tailscaled*.log 2>/dev/null | head -n1)
echo "logfile=$LOG"
[ -n "$LOG" ] && grep -iE 'ssh|SSH|port 65535|inbound' "$LOG" 2>/dev/null | tail -n25
echo "== tailscale ip for self =="
tailscale --socket=$S ip 2>&1 | head -n2
echo "== try tailscale ssh self ==="
tailscale --socket=$S ssh root@7a9405e8fb76 "hostname" 2>&1 | head -n3
echo "== serve listeners (from node) =="
tailscale --socket=$S serve status 2>&1 | head -n10
echo "== done =="
