#!/usr/bin/env bash
# run_v29_skelnet_v2.sh — 启动新一代 SkelNet-V2 (保幅校准+拓扑双分支+随机几何形变增强+阶段②接驳)
cd /root/Workspace/xy/DiT
mkdir -p logs
tmux kill-session -t v29_skelnet_v2 2>/dev/null || true

tmux new-session -d -s v29_skelnet_v2 "cd /root/Workspace/xy/DiT && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cu121/bin/python -m src.train.train --config src/train/configs/v29_skelnet_v2.json 2>&1 | tee /root/Workspace/xy/DiT/logs/v29_skelnet_v2.log"
echo "tmux session 'v29_skelnet_v2' successfully launched!"
