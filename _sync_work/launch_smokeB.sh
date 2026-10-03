#!/bin/bash
# 对照组 B: 同 ckpt 同 1k 步, w_style_rank=0
cd /root/Workspace/xy/DiT
CK=$(ls -t assets/results/v17_inj3_fixed_aug_100k/*/checkpoints/0100000.pt | head -1)
export PYTHONPATH=/root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
tmux kill-session -t styleB 2>/dev/null
tmux new-session -d -s styleB "$PY -u src/train/train.py --config src/train/configs/smoke_styleB.json --resume-full $CK 2>&1 | tee logs/_smoke_styleB.log"
echo "styleB launched"