#!/bin/bash
# gq 从 40000 续跑（唯一目的：让 poster 用上带「训练集类似条件 GT」两行的新 render_poster）。
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python
CKPT=assets/results/v17_gq_100k/20260924-185401-v17-gq-100k/checkpoints/0040000.pt
if pgrep -f 'src[.]train[.]train' >/dev/null; then
  echo "已有训练在跑, 不启动"; pgrep -af 'src[.]train[.]train' | cut -c1-120; exit 1
fi
[ -f "$CKPT.done" ] || { echo "✗ ckpt 未落盘: $CKPT"; exit 1; }
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== $(date '+%F %T') resume gq from 40000 ==="
exec $PY -u -m src.train.train --config src/train/configs/v17_gq_100k.json --resume-full "$CKPT"
