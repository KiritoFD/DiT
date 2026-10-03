#!/bin/bash
# start_s32b_daemon.sh — 启动 ctrl metrics daemon 指向 s32b (处理 eval_pending)
set -u
cd /root/Workspace/xy/DiT || exit 1
pkill -f 'eval_ctrl_metrics_daemon' 2>/dev/null
sleep 1
nohup /opt/conda/bin/python src/eval/eval_ctrl_metrics_daemon.py assets/results/s32b_repa_strong > /tmp/s32b_metrics_pipe.log 2>&1 &
echo "s32b ctrl metrics daemon pid=$!"
sleep 3
tail -5 /tmp/s32b_metrics_pipe.log