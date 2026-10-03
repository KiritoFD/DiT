#!/bin/bash
# start_gradio.sh — 启动 gradio_stdskel 服务 (CPU 推理, 带 --share)
# tmux 内直接写长命令会因 cwd/引号处理失败, 故用脚本包装。
# frpc 已手工安装: /opt/conda/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2 (v0.51.3)
set -u
cd /root/Workspace/xy/DiT || exit 1
export GRADIO_ANALYTICS_ENABLED=False
export PYTHONPATH=/root/Workspace/xy/DiT
exec /opt/conda/bin/python -u gradio_stdskel.py --device cpu --port 7863 --share
