#!/bin/bash
# inj3 的**修复数据版**：从 v17_inj3_100k 的 25000.pt resume 到 100k。
# 三件事(0.15 丢弃 / 拼进第一层 / 单层交叉注意力)不变；数据与评测全换成修复版。
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python
CKPT=assets/results/v17_inj3_100k/20260924-101345-v17-inj3-100k/checkpoints/0025000.pt

echo "=== $(date '+%F %T') 前提检查 ==="
if pgrep -f 'src[.]train[.]train' >/dev/null; then
  echo "已有训练在跑, 不启动"
  pgrep -af 'src[.]train[.]train' | cut -c1-140
  exit 1
fi
[ -f "$CKPT" ] || { echo "✗ resume ckpt 不存在: $CKPT"; exit 1; }
[ -f "$CKPT.done" ] || { echo "✗ $CKPT.done 缺失 —— ckpt 可能没写完"; exit 1; }
for f in assets/train_50k_v2_fixed.csv data/50k/shards_std_fixed \
         assets/eval_v13_strict_fixed.csv assets/eval_v13_seen_fixed.csv; do
  [ -e "$f" ] || { echo "✗ 缺文件: $f"; exit 1; }
done
echo "  resume ckpt  = $CKPT"
echo "  data_csv     = assets/train_50k_v2_fixed.csv"
echo "  skel shards  = data/50k/shards_std_fixed"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo "=== $(date '+%F %T') resume inj3-fixed -> 100k ==="
exec $PY -u -m src.train.train \
    --config src/train/configs/v17_inj3_fixed_100k.json \
    --resume-full "$CKPT"
