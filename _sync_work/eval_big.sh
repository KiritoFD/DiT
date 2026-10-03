#!/bin/bash
# 大容量版: 采样poster + dump(带 --base 64) + 下游实测
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
DS=data/top10_style23/skel64
CKPT=assets/skelnet_fm64_big.pt

echo "===== [1] 采样 poster (base 64) ====="
$PY -u tools/train_skelnet_fm64.py --poster 8 --resume $CKPT --dataset $DS \
    --base 64 --poster-out _ot_scratch/fm_big_poster.png 2>&1 \
    | grep -E 'poster\]|均值|^  [0-9]' | tail -12
$PY -u tools/poster_to_jpg.py 1000 _ot_scratch/fm_big_poster.png /tmp/fmbig.jpg 2>&1 | tail -1

echo "===== [2] dump ====="
$PY -u tools/train_skelnet_fm64.py --dump --resume $CKPT --dataset $DS \
    --base 64 --dump-width 3 --dump-tag big 2>&1 | grep -E '\[dump\]|Error'

echo "===== [3] 下游: 冻结 v26 ====="
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_fm_seen20_big \
    --pred-strict data/top10_style23/predskel_fm_strict84_big \
    --out assets/results/_calib_fm_big 2>&1 | grep -E 'set=.*pred ' | tail -2
echo "EVAL_BIG_DONE"
