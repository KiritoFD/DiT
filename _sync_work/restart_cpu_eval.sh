#!/bin/bash
# restart_cpu_eval.sh — 清旧协议产物 + 重启 daemon (幂等)
cd /root/Workspace/xy/DiT || exit 1
tmux kill-session -t cpu_eval 2>/dev/null
pkill -f cpu_eval_worker 2>/dev/null
pkill -f 'cpu_eval_daemon.py --mode' 2>/dev/null
sleep 2
D=$(ls -dt assets/results/v10a_skel_cond_pretrain/*/checkpoints 2>/dev/null | head -1)
if [ -n "$D" ]; then
  rm -f "$D"/eval_auto_*.json "$D"/*.cpu_eval.lock "$D"/.cpu_eval.lock 2>/dev/null
fi
tmux new-session -d -s cpu_eval 'export PYTHONPATH=/root/Workspace/xy/DiT; /opt/conda/envs/cu121/bin/python -u src/eval/cpu_eval_daemon.py --mode pretrain_g --watch-root /root/Workspace/xy/DiT/assets/results/v10a_skel_cond_pretrain --threads 32 > /tmp/cpu_eval_daemon.log 2>&1'
sleep 5
tmux has-session -t cpu_eval 2>/dev/null && echo CPU_EVAL_OK || { echo CPU_EVAL_FAIL; exit 1; }
tail -2 /tmp/cpu_eval_daemon.log
