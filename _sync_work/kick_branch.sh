#!/bin/bash
# 重启四臂分支 A/B (清干净旧进程 -> 重生成配置 -> tmux 起)
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
PY=/opt/conda/envs/cu121/bin/python

echo "=== 清旧 ==="
tmux kill-session -t branch 2>/dev/null && echo "tmux 已杀" || echo "无 tmux"
sleep 3
pkill -f 'v49_branch' && echo "旧训练已杀" || echo "无旧训练"
sleep 6
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "=== 重生成配置 ==="
sed -i 's/\r$//' tools/make_branch_configs.py _sync_work/run_branch_ab.sh
$PY tools/make_branch_configs.py 2>&1 | tail -7

echo "=== 起 tmux ==="
tmux new-session -d -s branch 'bash _sync_work/run_branch_ab.sh'
sleep 100
echo "=== 日志尾 ==="
grep -avE 'Warning|warn|pkg_resources|FutureWarning|_register_pytree' \
     exp-std/logs_branch/ab_latest.log 2>/dev/null | tail -14
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
