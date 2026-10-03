#!/bin/bash
# 干净基线（E0 复刻）。用途：第一步"恢复基线"——看 seen 是否回到 E0 曲线。
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python
if pgrep -f 'src[.]train[.]train' >/dev/null; then
  echo "已有训练在跑, 不启动"; pgrep -af 'src[.]train[.]train' | cut -c1-120; exit 1
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== $(date '+%F %T') v19 clean baseline 40k ==="
exec $PY -u -m src.train.train --config src/train/configs/v19_clean_baseline_40k.json
