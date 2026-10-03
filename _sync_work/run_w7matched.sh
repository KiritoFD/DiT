#!/bin/bash
# 等 shards_std_w7 造好 -> 自检守门 -> 开跑「等宽臂」K
#
# K 臂 = 条件 7px(std_w7) / 目标 7px(gtskel_w7) -> 任务退化成**纯形变(等宽)**。
# 对照: 旧 w7 臂是 (3px 条件 -> 7px 目标), 被迫同时加粗 2.3x -> resAlign 0.40。
# 而 G 臂(3px->3px, 纯形变) resAlign 0.49。所以等宽应当拿回这部分容量。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
ARMS=logs/skelnet_dit_arms
FILT='Warning|warn|pytree|pkg_resources|preload|skel-guard|eval'

while ! grep -q 'BUILD_STD_W7_DONE' /tmp/pivot.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] std_w7 就绪"
ls data/top10_style23/shards_std_w7/ | head -3

echo "===== [A] 守门自检: 旧错配配置(3px条件 -> 7px目标) 必须被拒 ====="
$PY -u tools/train_skelnet_dit.py --steps 2 --batch 4 --eval-every 100 --save-every 0 \
    --depth 6 --hidden 256 --heads 4 --inject-layers 2 \
    --bridge --bridge-hide-g \
    --cond-shards data/top10_style23/shards_std \
    --tgt-shards data/top10_style23/shards_gtskel_w7 \
    --log "$ARMS/guard_selftest.log" --out /tmp/guard_test.pt 2>&1 \
    | grep -E '宽度守门|FATAL|错配' | head -6
echo "    ^^^ 若上面出现 FATAL 错配 -> 守门有效"

echo "===== [B] K 臂: 条件 7px / 目标 7px (纯形变) ====="
$PY -u tools/train_skelnet_dit.py \
    --steps 30000 --batch 128 --lr 2e-4 --eval-every 1000 --save-every 500 \
    --val-n 128 --sample-steps 50 --es-patience 8 \
    --depth 6 --hidden 256 --heads 4 --inject-layers 2 \
    --bridge --bridge-hide-g \
    --cond-shards data/top10_style23/shards_std_w7 \
    --tgt-shards data/top10_style23/shards_gtskel_w7 \
    --loss-whiten --dir-weight 1.0 --mag-weight 1.0 --loss-ramp 1000 \
    --log "$ARMS/K_w7matched.log" --out assets/skelnet_dit_K_w7matched.pt 2>&1 \
    | grep -vE "$FILT"
echo K_DONE
