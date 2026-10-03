#!/bin/bash
# pixel-32 + AMP(bf16), 不 compile; 1 epoch = 训练集过 3 遍, 每 epoch 评测一次
set -u
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
DS=data/top10_style23/skel32
CKPT=assets/skelnet_fm_pix32.pt
M="--res 32 --dataset $DS --model DiT-2Cond-S/2 --glyph-inject-layers 4 --glyph-inject-mode adaln"

pkill -f 'train_skelnet_[f]m' 2>/dev/null
pkill -f 'eval_[p]ix32' 2>/dev/null
sleep 6
nvidia-smi --query-gpu=memory.used --format=csv,noheader

nohup $PY -u tools/train_skelnet_fm_dit.py $M \
    --epochs 30 --passes-per-epoch 3 --eval-every-epochs 1 --compile 0 \
    --batch 1024 --lr 3e-4 --wd 0.02 \
    --w-struct 0.5 --struct-min-t 0.4 --w-ink-ratio 0.3 \
    --val-n 128 --es-patience 4 \
    --log logs/skelnet_fm_pix32e.log --out $CKPT > /tmp/p32e.log 2>&1 &
echo "PID $!"
sleep 100
grep -E '1 epoch|参数|epoch |eval\]|Traceback|Error' logs/skelnet_fm_pix32e.log 2>/dev/null | head -8
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
