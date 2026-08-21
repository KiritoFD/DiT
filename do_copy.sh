#!/usr/bin/env bash
set -e
PUB="/c/Users/xy/.ssh/id_ed25519.pub"
if [ ! -f "$PUB" ]; then
  echo "NO PUB KEY AT $PUB"
  exit 1
fi
/usr/bin/ssh-copy-id -i "$PUB" -p 36430 -o StrictHostKeyChecking=accept-new root@10.176.54.17
echo "EXITCODE=$?"
