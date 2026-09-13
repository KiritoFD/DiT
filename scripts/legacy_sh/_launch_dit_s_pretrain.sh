#!/bin/bash
# Launch DiT-S 3Cond pretraining (full-parameter, from scratch) in tmux session `dit_s`.
# Usage: tmux new-session -d -s dit_s bash /root/Workspace/xy/DiT/_launch_dit_s_pretrain.sh
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/Workspace/xy/DiT
/opt/conda/bin/python train.py --config train_dit_s_pretrain.json > dit_s_pretrain_train.log 2>&1
echo "PRETRAIN_EXIT=$?"
