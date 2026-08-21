#!/bin/sh
S=/var/run/tailscale/tailscaled.sock
echo "== container pings archer-3 (100.126.104.49) =="
tailscale --socket=$S ping 100.126.104.49 2>&1 | head -n4
echo "== container status re archer-3 =="
tailscale --socket=$S status 2>&1 | grep -iE 'archer-3|100\.126\.104\.49'
echo "== container ping self =="
tailscale --socket=$S ping 100.72.205.57 2>&1 | head -n2
echo "== done =="
