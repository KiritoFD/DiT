#!/bin/bash
# GlyphQuery 版基模：从头训 100k。2 层骨架查询 @2,6 + 条件增强 + 8ch 拼接 + 0.15 丢弃。
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python

echo "=== $(date '+%F %T') 前提检查 ==="
if pgrep -f 'src[.]train[.]train' >/dev/null; then
  echo "已有训练在跑, 不启动"; pgrep -af 'src[.]train[.]train' | cut -c1-140; exit 1
fi
# 新模块必须在位（上一次会话只改了本地, 没同步 —— 这次已补）
[ -f src/model/glyph_query.py ] || { echo "✗ 缺 src/model/glyph_query.py"; exit 1; }
grep -q 'from .glyph_query import GlyphQuery' src/model/dit.py \
  || { echo "✗ dit.py 没接 GlyphQuery"; exit 1; }
for d in data/50k/shards_std_fixed data/50k/shards_std_aug_el1 \
         data/50k/shards_std_aug_el2 data/50k/shards_std_aug_aff \
         data/50k/shards_std_aug_drp; do
  [ -d "$d" ] || { echo "✗ 缺变体 shards: $d"; exit 1; }
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo "=== $(date '+%F %T') 从头训 v17-gq-100k ==="
exec $PY -u -m src.train.train --config src/train/configs/v17_gq_100k.json
