#!/bin/bash
# 链: 建 3px@64 数据集(两边同宽) -> FM 训练 -> dump -> 下游实测
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
DS=data/top10_style23/skel64_w3

echo "===== [1/4] 建 3px@64 数据集 ====="
$PY -u tools/build_skel64_dataset.py --res 64 --skel-width 3 --out $DS 2>&1 \
    | grep -E '落盘|墨占比|骨架为空|DONE'

echo "===== [2/4] FM 训练 (目标/输入 都是 3px@64) ====="
$PY -u tools/train_skelnet_fm64.py --dataset $DS \
    --steps 8000 --batch 2048 --base 32 --lr 4e-4 \
    --t-sampler logit_normal --sampler heun --sample-steps 50 \
    --eval-every 1000 --val-n 128 --es-patience 4 \
    --log logs/skelnet_fm64_w3.log --out assets/skelnet_fm64_w3.pt

echo "===== [3/4] dump (采样 -> 骨架化 -> 3px@256 -> encode) ====="
$PY -u tools/train_skelnet_fm64.py --dump --resume assets/skelnet_fm64_w3.pt \
    --dataset $DS --dump-width 3 --dump-tag w3 2>&1 | grep -E '\[dump\]|Error'

echo "===== [4/4] 下游: 冻结 v26 ====="
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_fm_seen20_w3 \
    --pred-strict data/top10_style23/predskel_fm_strict84_w3 \
    --out assets/results/_calib_fm_w3d 2>&1 | grep -E 'set=.*pred ' | tail -2
echo "CHAIN_W3_DONE"
