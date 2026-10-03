#!/bin/bash
# 低秩逐位置风格×几何乘法, 100k, cosine 与 E0 完全重合。
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python

echo "=== $(date '+%F %T') smoke ==="
$PY tools/smoke_lowrank_spatial.py || { echo "SMOKE FAILED"; exit 1; }

echo "=== $(date '+%F %T') gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
if pgrep -f 'src.train.train' >/dev/null; then
  echo "已有训练在跑, 不启动第二个"
  pgrep -af 'src.train.train' | cut -c1-140
  exit 1
fi

echo "=== $(date '+%F %T') train ==="
exec $PY -u -m src.train.train --config src/train/configs/v17_lowrank_spatial_r32.json
