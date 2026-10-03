#!/bin/bash
# 轮询等竞技场出 summary.txt (最多 ~14 分钟)
cd /root/Workspace/xy/DiT || exit 1
D=exp-std/signal_arena_full
L=exp-std/logs_purestd/arena_latest.log
for i in $(seq 1 14); do
  if [ -f "$D/summary.txt" ]; then echo "[done] summary.txt 出现"; break; fi
  echo "[$i] $(date '+%H:%M:%S')  产物: $(ls "$D" 2>/dev/null | tr '\n' ' ')  进程: $(ps -eo pid,cmd | grep -c '[l]atent_signal_arena')"
  sleep 60
done
echo
echo "########## 日志尾 30 行 ##########"
tail -30 "$L"
echo
echo "########## summary.txt ##########"
cat "$D/summary.txt" 2>/dev/null || echo "(还没有)"
echo
echo "########## 产物 ##########"
ls -la "$D"
echo "########## GPU ##########"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
