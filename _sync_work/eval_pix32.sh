#!/bin/bash
# 等 pixel-32 训练结束 -> poster -> dump -> 下游实测
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
DS=data/top10_style23/skel32
CKPT=assets/skelnet_fm_pix32.pt
M="--res 32 --dataset $DS --model DiT-2Cond-S/2 --glyph-inject-layers 4 --glyph-inject-mode adaln"

while pgrep -f "tools/train_skelnet_fm_dit.py --res 32" > /dev/null; do sleep 20; done
sleep 5
echo "=== 训练结束 ==="
grep -E 'eval\]|DONE' logs/skelnet_fm_pix32b.log | tail -12

echo "===== poster ====="
$PY -u tools/train_skelnet_fm_dit.py $M --poster 8 --resume $CKPT \
    --poster-out _ot_scratch/fm_pix32_poster.png 2>&1 | tail -13
$PY -u tools/poster_to_jpg.py 1000 _ot_scratch/fm_pix32_poster.png /tmp/pix32.jpg 2>&1 | tail -1

echo "===== dump ====="
$PY -u tools/train_skelnet_fm_dit.py $M --dump --resume $CKPT \
    --dump-width 3 --dump-tag p32 2>&1 | grep -E '\[dump\]|Error'

echo "===== 下游 ====="
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_fmdit_seen20_p32 \
    --pred-strict data/top10_style23/predskel_fmdit_strict84_p32 \
    --out assets/results/_calib_pix32 2>&1 | grep -E 'set=.*pred ' | tail -2
echo "EVAL_PIX32_DONE"
