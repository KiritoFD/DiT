#!/bin/bash
# 冒烟: c41x + 风格 token 每层注入 (style_token_n=32), 1000 步
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
NAME=smoke32

tmux kill-session -t "$NAME" 2>/dev/null
sleep 2
tmux new-session -d -s "$NAME" \
  "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; $PY -u src/train/train.py --config src/train/configs/smoke_c41x_sty32.json 2>&1 | tee /tmp/${NAME}_train.log"
sleep 12
echo "tmux:"; tmux ls 2>/dev/null | grep "$NAME"
echo "--- log ---"; tail -3 /tmp/${NAME}_train.log 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
