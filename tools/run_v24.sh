#!/usr/bin/env bash
# run_v24.sh — 拉起 v24_top10_style23 主干预训练 (150,000 steps)
cd /root/Workspace/xy/DiT
mkdir -p logs
tmux kill-session -t v24_top10 2>/dev/null || true

tmux new-session -d -s v24_top10 "cd /root/Workspace/xy/DiT && PYTHONPATH=. /opt/conda/envs/cu121/bin/python -m src.train.train --config src/train/configs/v24_top10_style23.json 2>&1 | tee /root/Workspace/xy/DiT/logs/v24_top10_style23.log"
echo "tmux session 'v24_top10' launched!"
