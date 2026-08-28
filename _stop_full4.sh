#!/bin/bash
# Stop s17 + serial chain + eval daemons, free GPU
tmux kill-session -t full4 2>/dev/null
sleep 2
# Kill any remaining train.py / daemon processes
pkill -f "train.py --config s17_s_flow.json" 2>/dev/null
pkill -f "eval_metrics_daemon.py" 2>/dev/null
pkill -f "run_4full.sh" 2>/dev/null
sleep 3
echo "=== processes after kill ==="
ps aux | grep -E "train.py|eval_metrics|run_4full" | grep -v grep | head
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== tmux ==="
tmux ls 2>&1