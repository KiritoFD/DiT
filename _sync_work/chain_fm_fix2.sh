#!/bin/bash
# 链 v2: 结构 loss 门控改为 t>=0.4 (高噪声段, 真正考结构)
set -u
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
DS=data/top10_style23/skel64

pkill -f 'train_skelnet_[f]m' 2>/dev/null
sleep 5

echo "===== [1/3] FM 训练 (结构 loss 门控 t>=0.4) ====="
$PY -u tools/train_skelnet_fm64.py --dataset $DS \
    --steps 10000 --batch 2048 --base 32 --lr 4e-4 \
    --t-sampler logit_normal --sampler heun --sample-steps 50 \
    --w-struct 0.5 --struct-min-t 0.4 --ink-w 10 \
    --eval-every 1000 --val-n 128 --es-patience 5 \
    --log logs/skelnet_fm64_fix2.log --out assets/skelnet_fm64_fix2.pt

echo "===== [2/3] dump ====="
$PY -u tools/train_skelnet_fm64.py --dump --resume assets/skelnet_fm64_fix2.pt \
    --dataset $DS --dump-width 3 --dump-tag fix2 2>&1 | grep -E '\[dump\]|Error'

echo "===== [3/3] 下游: 冻结 v26 ====="
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_fm_seen20_fix2 \
    --pred-strict data/top10_style23/predskel_fm_strict84_fix2 \
    --out assets/results/_calib_fm_fix2 2>&1 | grep -E 'set=.*pred ' | tail -2
echo "CHAIN_FM_FIX2_DONE"
