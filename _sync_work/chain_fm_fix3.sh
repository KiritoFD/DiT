#!/bin/bash
# 链 v3: 去掉 ink-w(过墨推手) + 加墨量约束; 结构 loss 门控 t>=0.4
set -u
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
DS=data/top10_style23/skel64

pkill -f 'train_skelnet_[f]m' 2>/dev/null
sleep 5

echo "===== [1/3] FM 训练 (ink-w=0, +墨量约束 w=0.3) ====="
$PY -u tools/train_skelnet_fm64.py --dataset $DS \
    --steps 6000 --batch 2048 --base 32 --lr 4e-4 \
    --t-sampler logit_normal --sampler heun --sample-steps 50 \
    --w-struct 0.5 --struct-min-t 0.4 --ink-w 0 --w-ink-ratio 0.3 \
    --eval-every 500 --val-n 128 --es-patience 4 \
    --log logs/skelnet_fm64_fix3.log --out assets/skelnet_fm64_fix3.pt

echo "===== [2/3] dump ====="
$PY -u tools/train_skelnet_fm64.py --dump --resume assets/skelnet_fm64_fix3.pt \
    --dataset $DS --dump-width 3 --dump-tag fix3 2>&1 | grep -E '\[dump\]|Error'

echo "===== [3/3] 下游: 冻结 v26 ====="
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_fm_seen20_fix3 \
    --pred-strict data/top10_style23/predskel_fm_strict84_fix3 \
    --out assets/results/_calib_fm_fix3 2>&1 | grep -E 'set=.*pred ' | tail -2
echo "CHAIN_FM_FIX3_DONE"
