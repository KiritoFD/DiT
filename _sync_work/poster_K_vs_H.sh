#!/bin/bash
# 同 6 个样本 / 同列, 出两张骨架 poster: K 臂(5px条件) vs H 参考臂(3px条件)
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
FILT='Warning|warn|pytree|pkg_resources|preload|skel-guard'
COMMON="--n 6 --stride 400"

echo "===== K 臂 (条件 5px = shards_std_w7) ====="
$PY -u tools/poster_skelnet_now.py $COMMON \
    --resume assets/skelnet_dit_K_w7matched.pt \
    --std-dir data/top10_style23/shards_std_w7 \
    --out _ot_scratch/poster_K.png 2>&1 | grep -vE "$FILT" | tail -14

echo "===== H 参考臂 (条件 3px = shards_std) ====="
$PY -u tools/poster_skelnet_now.py $COMMON \
    --resume assets/skelnet_dit_H_bridge_nog_w7.pt.best \
    --out _ot_scratch/poster_H.png 2>&1 | grep -vE "$FILT" | tail -14
echo POSTERS_DONE
