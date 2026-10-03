#!/bin/bash
# 重建 std 骨架到**与目标同宽** -> 开 K 臂
#
# 探针事实 (iters=3 / 7px 时):
#   std 原样(3px) 墨占比 0.03667 | 膨胀3次(7px) 0.08996 | 目标 gtskel_w7 0.06540
#   -> 目标实际宽度 ≈ 5.2px (不是 7px!); 膨胀3次比目标粗 1.38x
#   -> 改用 iters=2 (5px), 预期墨占比 ~0.063 (与目标 0.065 对齐)
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
FILT='Warning|warn|pytree|pkg_resources|preload|skel-guard'

echo "======== 探针 iters=2 (5px) ========"
$PY -u tools/build_std_w7.py --probe --limit 1 --iters 2 2>&1 | grep -vE "$FILT" | tail -10

echo "======== 重建 shards_std_w7 (5px) ========"
rm -rf data/top10_style23/shards_std_w7
$PY -u tools/build_std_w7.py --iters 2 --batch 32 2>&1 | grep -vE "$FILT" | tail -14
echo REBUILD5_DONE

echo "======== K 臂: 条件 5px / 目标 5px (纯形变) ========"
$PY -u tools/train_skelnet_dit.py \
    --steps 30000 --batch 128 --lr 2e-4 --eval-every 1000 --save-every 500 \
    --val-n 128 --sample-steps 50 --es-patience 8 \
    --depth 6 --hidden 256 --heads 4 --inject-layers 2 \
    --bridge --bridge-hide-g \
    --cond-shards data/top10_style23/shards_std_w7 \
    --tgt-shards data/top10_style23/shards_gtskel_w7 \
    --loss-whiten --dir-weight 1.0 --mag-weight 1.0 --loss-ramp 1000 \
    --log logs/skelnet_dit_arms/K_w7matched.log \
    --out assets/skelnet_dit_K_w7matched.pt 2>&1 | grep -vE "$FILT"
echo K_DONE
