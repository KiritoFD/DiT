#!/usr/bin/env bash
# Ephemeral forwarded agent: server-to-server transfer without copying private keys.
set -euo pipefail
eval "$(ssh-agent -s)"
trap 'ssh-agent -k >/dev/null' EXIT
ssh-add /c/Users/xy/.ssh/id_ed25519
ssh -A -o BatchMode=yes ds@10.222.120.101 \
  'scp -o BatchMode=yes -o StrictHostKeyChecking=yes -P 36430 root@10.176.54.17:/root/Workspace/xy/DiT/assets/qwen_lora_20261023/data.tar.gz /home/ds/Workspace/xy/qwen-lora-20261023/data.tar.gz'
