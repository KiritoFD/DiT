#!/bin/bash
# 停训练 -> 出 predskel 采样 poster -> 转 jpg
set -u
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
DS=data/top10_style23/skel32
CKPT=assets/skelnet_fm_pix32.pt
M="--res 32 --dataset $DS --model DiT-2Cond-S/2 --glyph-inject-layers 4 --glyph-inject-mode adaln --compile 0"

pkill -f 'train_skelnet_[f]m' 2>/dev/null
sleep 8
echo "=== best ckpt ==="
ls -la $CKPT
echo "=== eval 历史 ==="
grep -E 'eval\]' logs/skelnet_fm_pix32e.log | tail -10

echo "=== poster (predskel 采样) ==="
$PY -u tools/train_skelnet_fm_dit.py $M --poster 10 --resume $CKPT \
    --poster-out _ot_scratch/fm_pix32_poster.png 2>&1 | tail -16
$PY -u tools/poster_to_jpg.py 1000 _ot_scratch/fm_pix32_poster.png /tmp/pix32.jpg 2>&1 | tail -1
