#!/bin/bash
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# kill any stale s13 process
pkill -f "train.py --config s13_3top30_dino_xs" 2>/dev/null
sleep 2
# fresh log
rm -f /tmp/s13_train.log
tmux new-session -d -s s13_train "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128; cd /root/Workspace/xy/DiT && /opt/conda/bin/python train.py --config s13_3top30_dino_xs.json > /tmp/s13_train.log 2>&1"
sleep 5
echo "--- proc ---"
pgrep -f "train.py --config s13_3top30_dino_xs" | head -2
echo "--- log head ---"
head -5 /tmp/s13_train.log
