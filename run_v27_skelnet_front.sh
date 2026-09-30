#!/usr/bin/env bash
# run_v27_skelnet_front.sh — 阶段②: 冻结 v26(GT骨架模型), 把 SkelNet 接到前面只训 SkelNet
#
# 链路:  g_std --SkelNet(可训)--> g' --> v26 主干(冻结) --> 生成
# 用法:  V26_CKPT=<path> bash run_v27_skelnet_front.sh
cd /root/Workspace/xy/DiT
mkdir -p logs
tmux kill-session -t v27_skelnet 2>/dev/null || true

V26_CKPT=${V26_CKPT:-assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt}
echo "V26_CKPT = $V26_CKPT"

tmux new-session -d -s v27_skelnet "cd /root/Workspace/xy/DiT && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cu121/bin/python -m src.train.train --config src/train/configs/v27_skelnet_front.json --resume-full $V26_CKPT 2>&1 | tee /root/Workspace/xy/DiT/logs/v27_skelnet.log"
echo "tmux session 'v27_skelnet' launched!  tail -f logs/v27_skelnet.log"
