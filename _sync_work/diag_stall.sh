#!/bin/bash
echo "=== 进程 ==="
ps -eo pid,etime,pcpu,cmd | grep -E '[t]rain_skelnet|[c]hain_fm' | head -6
echo
echo "=== chain 日志 ==="
tail -12 /tmp/chain_fm_fix.log 2>/dev/null
echo
echo "=== 训练日志 (最后 20 行) ==="
tail -20 /root/Workspace/xy/DiT/logs/skelnet_fm64_fix.log 2>/dev/null
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -4
