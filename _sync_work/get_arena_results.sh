#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1
D=exp-std/signal_arena_full
L=exp-std/logs_purestd/arena_latest.log
echo "=== 现在 $(date '+%F %T') ==="
echo "=== 进程 ==="; ps -eo pid,etime,pcpu,cmd | grep '[l]atent_signal_arena' | head -2
echo "=== GPU ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo
echo "=== 日志(去噪) 全部 ==="
grep -vE 'Warning|warn|pkg_resources|FutureWarning|_register_pytree|warnings.warn' "$L" 2>/dev/null | tail -60

echo
echo "=== summary.txt ==="
cat "$D/summary.txt" 2>/dev/null || echo "(还没有)"

echo
echo "=== margin.csv ==="
cat "$D/margin.csv" 2>/dev/null || echo "(还没有)"

echo
echo "=== inversion.csv ==="
cat "$D/inversion.csv" 2>/dev/null || echo "(还没有)"

echo
echo "=== 产物 ==="
ls -la "$D" 2>/dev/null
