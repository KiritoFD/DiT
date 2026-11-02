#!/bin/bash
# Pull the key checkpoints once the VPN is back. Idempotent: safe to re-run after a drop.
#
#   bash tools/p0_s4_ft/pull_ckpts.sh          # all four
#   bash tools/p0_s4_ft/pull_ckpts.sh K,S4     # a subset
#
# Notes for whoever runs this:
#   * S5 lives on the 48 box, so it needs 48 reachable as well as the 4090.
#   * ~7.2 GB total at ~440 KB/s uplink from 4090 is roughly 4.5 h; rsync resumes, so a
#     mid-transfer VPN drop costs only the bytes in flight.
#   * Verification is a real torch.load followed by a SHA256 compare. A file with the
#     right byte count but wrong contents is rejected.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY=${PY:-python}
ONLY=${1:-}

echo "=== connectivity pre-check ==="
ok=1
for h in 4090 48; do
  if ssh -o ConnectTimeout=10 -o BatchMode=yes "$h" 'true' 2>/dev/null; then
    echo "  $h: reachable"
  else
    echo "  $h: UNREACHABLE"
    ok=0
  fi
done
if [ "$ok" = 0 ]; then
  echo
  echo "Not all hosts reachable. Likely the aTrust VPN is down:"
  echo "  ipconfig | grep -A3 aTrust     # expect '媒体状态 . . . : 已连接'"
  echo "Reconnect it, then re-run this script."
  echo "Continuing anyway would only fail later; nothing has been transferred."
  exit 1
fi

echo
if [ -n "$ONLY" ]; then
  $PY tools/p0_s4_ft/pull_ckpts.py --only "$ONLY"
else
  $PY tools/p0_s4_ft/pull_ckpts.py
fi
