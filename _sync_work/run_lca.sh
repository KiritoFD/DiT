#!/usr/bin/env bash
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python
echo "=== $(date '+%T') stop previous ==="
for p in $(pgrep -f 'src.train.train'); do
  kill "$p" 2>/dev/null || true
done
sleep 8
if pgrep -f 'src.train.train' >/dev/null; then
  echo "停不掉"
  exit 1
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== $(date '+%T') train ==="
exec $PY -u -m src.train.train --config src/train/configs/v17_lca_x.json
