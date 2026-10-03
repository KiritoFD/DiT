#!/bin/bash
# run_patch1.sh — 主推升级: 把生成模型的**结构分辨率**打开 (patch 2 -> 1)
#
# 依据:
#  · 模型本来就是生成模型 (DiT_2Cond + rectified flow, bridge 起步, hide-g),
#    问题不是"判别 vs 生成", 而是 patch=2 把 32² latent 压成 16²=256 token,
#    3px 骨架在 latent 里只占 0.375 格 -> 细结构没有承载位置。
#  · K 臂(等宽 5px 条件)被 poster 判死: 6/6 行"照抄输入" -> 已弃。
#  · 保留 H 的设定 (3px 条件 -> w7 目标) + 方向/模长项 (防照抄/防幅度收缩)。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
ARMS=logs/skelnet_dit_arms
FILT='Warning|warn|pytree|pkg_resources|preload|skel-guard|eval'

echo "===== L 臂: patch=1 (1024 token) + 3px 条件 -> w7 目标 + 方向/模长 ====="
$PY -u tools/train_skelnet_dit.py \
    --steps 30000 --batch 64 --lr 2e-4 --eval-every 1000 --save-every 500 \
    --val-n 128 --sample-steps 50 --es-patience 8 \
    --depth 6 --hidden 256 --heads 4 --patch 1 \
    --bridge --bridge-hide-g \
    --cond-shards data/top10_style23/shards_std \
    --tgt-shards data/top10_style23/shards_gtskel_w7 \
    --loss-whiten --dir-weight 1.0 --mag-weight 1.0 --loss-ramp 1000 \
    --log "$ARMS/L_patch1.log" --out assets/skelnet_dit_L_patch1.pt 2>&1 \
    | grep -vE "$FILT" | tail -6
echo L_DONE

echo "===== M 臂: patch=1 + 更大容量 (depth 10 / hidden 384) ====="
$PY -u tools/train_skelnet_dit.py \
    --steps 30000 --batch 48 --lr 2e-4 --eval-every 1000 --save-every 500 \
    --val-n 128 --sample-steps 50 --es-patience 8 \
    --depth 10 --hidden 384 --heads 6 --patch 1 \
    --bridge --bridge-hide-g \
    --cond-shards data/top10_style23/shards_std \
    --tgt-shards data/top10_style23/shards_gtskel_w7 \
    --loss-whiten --dir-weight 1.0 --mag-weight 1.0 --loss-ramp 1000 \
    --log "$ARMS/M_patch1_big.log" --out assets/skelnet_dit_M_patch1_big.pt 2>&1 \
    | grep -vE "$FILT" | tail -6
echo M_DONE
