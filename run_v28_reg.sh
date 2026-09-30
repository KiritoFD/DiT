#!/usr/bin/env bash
# run_v28_reg.sh — v28-reg: v27 + 抗过拟合 (正则化/res_cap/WD/输入加噪) + 书家表 cross-attn
# 用法: bash run_v28_reg.sh
cd /root/Workspace/xy/DiT
mkdir -p logs
tmux kill-session -t v28_reg 2>/dev/null || true

V26_CKPT=${V26_CKPT:-assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt}
echo "V26_CKPT = $V26_CKPT"

tmux new-session -d -s v28_reg "cd /root/Workspace/xy/DiT && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cu121/bin/python -m src.train.train --config src/train/configs/v28_reg.json --resume-full $V26_CKPT 2>&1 | tee /root/Workspace/xy/DiT/logs/v28_reg.log"
echo "tmux session 'v28_reg' launched!  tail -f logs/v28_reg.log"
