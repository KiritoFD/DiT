#!/bin/bash
# 启动 s20 controlnet 后训练 (warm-start, 冻结主模型) —— v2: 已对齐 skel latent + xformers
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# 清理旧会话/进程
tmux kill-session -t ctrl_s20 2>/dev/null
pkill -f "train_controlnet --config src/train/configs/s20_ctrl_skel_flow_v2" 2>/dev/null
sleep 2
rm -f /tmp/s20_ctrl_train.log
tmux new-session -d -s ctrl_s20 "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128; cd /root/Workspace/xy/DiT && /opt/conda/bin/python -m src.train.train_controlnet --config src/train/configs/s20_ctrl_skel_flow_v2.json > /tmp/s20_ctrl_train.log 2>&1"
sleep 15
echo "--- tmux ---"
tmux ls 2>/dev/null
echo "--- proc ---"
pgrep -f "train_controlnet --config src/train/configs/s20_ctrl_skel_flow_v2" | head -2
echo "--- log head ---"
head -60 /tmp/s20_ctrl_train.log 2>/dev/null
