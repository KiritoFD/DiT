#!/bin/sh
S=/var/run/tailscale/tailscaled.sock
echo "== status json (ssh/running/auth) =="
tailscale --socket=$S status --json 2>/dev/null | tr ',' '\n' | grep -iE '"SSHEnabled"|"Running"|"Online"|"WantRunning"' | head
echo "== prefs =="
tailscale --socket=$S debug prefs 2>/dev/null | tr ',' '\n' | grep -iE 'ssh|wantrunning|tun' | head
echo "== done =="
