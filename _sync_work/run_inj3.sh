#!/bin/bash
# 三件事一次做: 训练时骨架 0.15 丢弃 + 骨架拼进第一层卷积(8ch, 新通道 0 init) + 只留第 6 层交叉注意力(风格缩放夹 [-1,1], 残差固定 0.1)。
# 100k, 稳定主干, 无中程监督。
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python

echo "=== $(date '+%F %T') 检查 GPU 是否空闲 ==="
if pgrep -f 'src.train.train' >/dev/null; then
  echo "已有训练在跑, 不启动"
  pgrep -af 'src.train.train' | cut -c1-140
  exit 1
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo "=== $(date '+%F %T') train inj3 100k ==="
exec $PY -u -m src.train.train --config src/train/configs/v17_inj3_100k.json
