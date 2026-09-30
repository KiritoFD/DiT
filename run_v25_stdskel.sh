#!/usr/bin/env bash
# run_v25_stdskel.sh — v25-stdskel: Top10 纯 std-skel 直通 (彻底移除 SkelNet) 150k 训练
#
# 用途: 作为 SkelNet 的**消融基线** —— 条件 g 直接取标准字骨架 g_std (来自
#       data/top10_style23/shards_std), 不经过任何形变网络。
# 与 v24_frozenskel 的唯一差异: deform_skel = 0 (无 SkelNet / 无 deform_ckpt /
#       inst_skel_shards_dir 空 / w_deform_skel = 0)。
#
# ⚠ 启动前确认 GPU 空闲 (本脚本不检查; 若他人任务在跑会 OOM)。
cd /root/Workspace/xy/DiT
mkdir -p logs
tmux kill-session -t v25_stdskel 2>/dev/null || true

tmux new-session -d -s v25_stdskel "cd /root/Workspace/xy/DiT && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cu121/bin/python -m src.train.train --config src/train/configs/v25_stdskel.json 2>&1 | tee /root/Workspace/xy/DiT/logs/v25_stdskel.log"
echo "tmux session 'v25_stdskel' launched!  tail -f logs/v25_stdskel.log"
