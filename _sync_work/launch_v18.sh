#!/bin/bash
# v18_style_rank_200k 从头预训练 (tmux v18)
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
TMUX=stylesweep
tmux kill-session -t $TMUX 2>/dev/null
tmux kill-session -t rankdino 2>/dev/null
tmux kill-session -t styleA 2>/dev/null
tmux kill-session -t styleB 2>/dev/null
TS=$(date +%Y%m%d-%H%M%S)
LOG=logs/v18_series/v18_train_$TS.log
mkdir -p logs/v18_series
tmux new-session -d -s v18 "$PY -u src/train/train.py --config src/train/configs/v18_style_rank_200k.json 2>&1 | tee $LOG"
echo "[v18] launched (tmux v18). log: $LOG"
echo "[v18] watch: tail -f $LOG"