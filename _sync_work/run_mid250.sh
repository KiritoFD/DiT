#!/usr/bin/env bash
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python
if pgrep -f 'src.train.train' >/dev/null; then
  echo "已有训练在跑"
  pgrep -af 'src.train.train' | cut -c1-140
  exit 1
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== $(date '+%T') train mid 250k ==="
exec $PY -u -m src.train.train --config src/train/configs/v17_mid_250k.json
