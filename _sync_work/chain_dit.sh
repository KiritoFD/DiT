#!/bin/bash
# 链: DiT-2Cond-S/2 (照抄 v25/v26) 在 64² 像素域做 flow -> poster -> dump -> 下游
set -u
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
DS=data/top10_style23/skel64
CKPT=assets/skelnet_fm_dit.pt

pkill -f 'train_skelnet_[f]m' 2>/dev/null
sleep 5

echo "===== [1/4] 训练 DiT-2Cond-S/2 in pixel space ====="
$PY -u tools/train_skelnet_fm_dit.py --dataset $DS \
    --model DiT-2Cond-S/2 --glyph-inject-layers 4 --glyph-inject-mode adaln \
    --steps 2500 --batch 256 --lr 1e-4 --wd 0.02 \
    --w-struct 0.5 --struct-min-t 0.4 --w-ink-ratio 0.3 \
    --eval-every 250 --val-n 128 --es-patience 6 \
    --log logs/skelnet_fm_dit.log --out $CKPT

echo "===== [2/4] poster ====="
$PY -u tools/train_skelnet_fm_dit.py --poster 8 --resume $CKPT --dataset $DS \
    --model DiT-2Cond-S/2 --glyph-inject-layers 4 --glyph-inject-mode adaln \
    --poster-out _ot_scratch/fm_dit_poster.png 2>&1 | tail -13
$PY -u tools/poster_to_jpg.py 1000 _ot_scratch/fm_dit_poster.png /tmp/fmdit.jpg 2>&1 | tail -1

echo "===== [3/4] dump ====="
$PY -u tools/train_skelnet_fm_dit.py --dump --resume $CKPT --dataset $DS \
    --model DiT-2Cond-S/2 --glyph-inject-layers 4 --glyph-inject-mode adaln \
    --dump-width 3 --dump-tag dit 2>&1 | grep -E '\[dump\]|Error'

echo "===== [4/4] 下游: 冻结 v26 ====="
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_fmdit_seen20_dit \
    --pred-strict data/top10_style23/predskel_fmdit_strict84_dit \
    --out assets/results/_calib_fmdit 2>&1 | grep -E 'set=.*pred ' | tail -2
echo "CHAIN_DIT_DONE"
