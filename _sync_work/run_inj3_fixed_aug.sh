#!/bin/bash
# inj3-fixed + 全部条件增强。从 v17_inj3_fixed_100k 的 55000.pt resume 到 100k。
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python
CKPT=assets/results/v17_inj3_fixed_100k/20260924-121135-v17-inj3-fixed-100k/checkpoints/0055000.pt

echo "=== $(date '+%F %T') 前提检查 ==="
if pgrep -f 'src[.]train[.]train' >/dev/null; then
  echo "已有训练在跑, 不启动"; pgrep -af 'src[.]train[.]train' | cut -c1-140; exit 1
fi
[ -f "$CKPT" ] || { echo "✗ resume ckpt 不存在: $CKPT"; exit 1; }
[ -f "$CKPT.done" ] || { echo "✗ $CKPT.done 缺失"; exit 1; }

# 5 个骨架变体目录必须齐全（缺一个 -> 数据集启动即报错，这里提前拦）
for d in data/50k/shards_std_fixed \
         data/50k/shards_std_aug_el1 data/50k/shards_std_aug_el2 \
         data/50k/shards_std_aug_aff data/50k/shards_std_aug_drp; do
  n=$(ls "$d"/shard_*.npz 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then echo "✗ 缺变体 shards: $d"; exit 1; fi
  echo "  $d : $n shard"
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo "=== $(date '+%F %T') resume inj3-fixed-aug -> 100k ==="
exec $PY -u -m src.train.train \
    --config src/train/configs/v17_inj3_fixed_aug_100k.json \
    --resume-full "$CKPT"
