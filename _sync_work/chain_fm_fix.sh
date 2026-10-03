#!/bin/bash
# 链: FM(带结构 loss + 瓶颈注意力) 训练 -> dump -> 下游实测 -> poster
# 保持 1px@64 同宽度表述不变, 只修 loss
set -u
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
DS=data/top10_style23/skel64

pkill -f 'train_skelnet_[f]m' 2>/dev/null
sleep 5

echo "===== [1/3] FM 训练 (纯 MSE -> MSE + 结构 loss) ====="
$PY -u tools/train_skelnet_fm64.py --dataset $DS \
    --steps 10000 --batch 2048 --base 32 --lr 4e-4 \
    --t-sampler logit_normal --sampler heun --sample-steps 50 \
    --w-struct 0.3 --struct-max-t 0.3 \
    --eval-every 1000 --val-n 128 --es-patience 5 \
    --log logs/skelnet_fm64_fix.log --out assets/skelnet_fm64_fix.pt

echo "===== [2/3] dump ====="
$PY -u tools/train_skelnet_fm64.py --dump --resume assets/skelnet_fm64_fix.pt \
    --dataset $DS --dump-width 3 --dump-tag fix 2>&1 | grep -E '\[dump\]|Error'

echo "===== [3/3] 下游: 冻结 v26 ====="
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_fm_seen20_fix \
    --pred-strict data/top10_style23/predskel_fm_strict84_fix \
    --out assets/results/_calib_fm_fix 2>&1 | grep -E 'set=.*pred ' | tail -2
echo "CHAIN_FM_FIX_DONE"
