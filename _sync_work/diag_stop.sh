#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1
echo "=== 现在 $(date '+%F %T') ==="
echo "=== tmux ==="; tmux ls 2>&1 | head -3
echo "=== 进程 ==="; ps -eo pid,etime,pcpu,rss,cmd | grep -E 'encode_flux|latent_signal|flux_pipeline' | grep -v grep | head -6
echo "=== GPU ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
echo "=== GPU 进程 ==="; nvidia-smi --query-compute-apps=pid,used_memory --format=csv | head -4
echo
echo "=== 日志尾 40 行 ==="; tail -40 exp-std/logs_purestd/flux_pipeline_latest.log 2>/dev/null
echo
echo "=== 日志里的错误 ==="; grep -nE 'Error|error|Traceback|assert|EXIT|失败|No such' exp-std/logs_purestd/flux_pipeline_latest.log 2>/dev/null | tail -12
echo
echo "=== 产物 ==="
for d in exp-std/data/shards_img_flux16 exp-std/data/shards_std_flux16 exp-std/signal_arena_full; do
  echo "--- $d"; ls -la "$d" 2>/dev/null | tail -5
done
echo "=== std 源目录 ==="; ls data/top10_style23/ 2>/dev/null | head -12
echo -n "std png 数 = "; ls data/top10_style23/std/*.png 2>/dev/null | wc -l
echo -n "imgs png 数 = "; ls data/top10_style23/imgs/*.png 2>/dev/null | wc -l
echo "=== 磁盘 ==="; df -h /root | tail -1
