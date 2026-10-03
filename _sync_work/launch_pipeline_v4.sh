#!/bin/bash
# 启动/重启 pipeline v4 (幂等: 已有 ckpt 的阶段自动跳过)
set -u
cd /root/Workspace/xy/DiT || exit 1
tmux kill-session -t s30pipe 2>/dev/null
sleep 1
tmux new-session -d -s s30pipe 'bash _sync_work/run_s30_skel_pipeline_v4.sh'
sleep 2
tmux ls
echo "pipeline v4 started in tmux s30pipe"