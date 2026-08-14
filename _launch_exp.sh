#!/bin/bash
# 通用实验启动脚本：bash _launch_exp.sh <config.json> <logfile>
# 在 tmux 会话 `exp` 中后台运行，日志写到独立文件。
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/Workspace/xy/DiT
/opt/conda/bin/python train.py --config "$1" > "$2" 2>&1
echo "EXIT=$?"
