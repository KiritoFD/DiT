#!/bin/bash
# restart_ctrl_daemon.sh — 重启 s31 ctrl metrics daemon (幂等)
set -u
cd /root/Workspace/xy/DiT || exit 1
pkill -f 'eval_ctrl_metrics_daemon' 2>/dev/null
sleep 1
nohup /opt/conda/bin/python src/eval/eval_ctrl_metrics_daemon.py assets/results/s31_ctrl_gt_skel_1px > /tmp/s31_metrics_pipe.log 2>&1 &
echo "ctrl metrics daemon restarted, pid=$!"
sleep 3
tail -5 /tmp/s31_metrics_pipe.log