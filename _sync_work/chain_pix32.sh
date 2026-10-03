#!/bin/bash
# pixel-32 路线: 简单下采样代替 VAE, S/2 (256 token 设计点)
set -u
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
DS=data/top10_style23/skel32
CKPT=assets/skelnet_fm_pix32.pt
MODEL=DiT-2Cond-S/2

pkill -f 'train_skelnet_[f]m' 2>/dev/null
sleep 5

echo "===== [0/4] 建 32² 数据集 ====="
$PY -u tools/build_skel64_dataset.py --res 32 --out $DS 2>&1 \
    | grep -E '落盘|墨占比|骨架为空|DONE'

echo "===== [1/4] 训练 DiT-S/2 @ 32² ====="
$PY -u tools/train_skelnet_fm_dit.py --res 32 --dataset $DS \
    --model $MODEL --glyph-inject-layers 4 --glyph-inject-mode adaln \
    --steps 4000 --batch 512 --lr 2e-4 --wd 0.02 \
    --w-struct 0.5 --struct-min-t 0.4 --w-ink-ratio 0.3 \
    --eval-every 500 --val-n 128 --es-patience 6 \
    --log logs/skelnet_fm_pix32.log --out $CKPT

echo "===== [2/4] poster ====="
$PY -u tools/train_skelnet_fm_dit.py --res 32 --dataset $DS \
    --model $MODEL --glyph-inject-layers 4 --glyph-inject-mode adaln \
    --poster 8 --resume $CKPT --poster-out _ot_scratch/fm_pix32_poster.png 2>&1 | tail -13
$PY -u tools/poster_to_jpg.py 1000 _ot_scratch/fm_pix32_poster.png /tmp/pix32.jpg 2>&1 | tail -1

echo "===== [3/4] dump ====="
$PY -u tools/train_skelnet_fm_dit.py --res 32 --dataset $DS \
    --model $MODEL --glyph-inject-layers 4 --glyph-inject-mode adaln \
    --dump --resume $CKPT --dump-width 3 --dump-tag p32 2>&1 | grep -E '\[dump\]|Error'

echo "===== [4/4] 下游 ====="
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_fmdit_seen20_p32 \
    --pred-strict data/top10_style23/predskel_fmdit_strict84_p32 \
    --out assets/results/_calib_pix32 2>&1 | grep -E 'set=.*pred ' | tail -2
echo "CHAIN_PIX32_DONE"
