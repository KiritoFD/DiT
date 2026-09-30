#!/usr/bin/env bash
# run_v24_frozenskel.sh — 完全冻结 SkelNet 的 v24-frozenskel 150k 训练
cd /root/Workspace/xy/DiT
mkdir -p logs
tmux kill-session -t v24_frozenskel 2>/dev/null || true

tmux new-session -d -s v24_frozenskel "cd /root/Workspace/xy/DiT && PYTHONPATH=. /opt/conda/envs/cu121/bin/python -m src.train.train --config src/train/configs/v24_frozenskel.json 2>&1 | tee /root/Workspace/xy/DiT/logs/v24_frozenskel.log"
echo "tmux session 'v24_frozenskel' launched!"
