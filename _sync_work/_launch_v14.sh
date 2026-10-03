#!/bin/bash
# v14 full-ft: 从 stage2@60k ckpt 全解冻重训
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
CKPT=assets/results/v14_style87_s2/20260918-234236-v14-style87-stage2/checkpoints/0060000.pt
nohup /opt/conda/envs/cu121/bin/python -u src/train/train.py \
    --config src/train/configs/v14_style87_fullft.json \
    --resume-full "$CKPT" \
    > /tmp/v14_fullft.log 2>&1 &
echo "[full-ft] started, log=/tmp/v14_fullft.log"



