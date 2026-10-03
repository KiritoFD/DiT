#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1
echo "=== tmux ==="; tmux ls 2>&1 | head -4
echo "=== 进程 ==="; ps -eo pid,etime,pcpu,cmd | grep -E 'encode_flux|latent_signal|flux_pipeline' | grep -v grep | head -6
echo "=== GPU ==="; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
echo "=== GPU 进程 ==="; nvidia-smi --query-compute-apps=pid,used_memory --format=csv | head -5
echo "=== 日志尾 20 行 ==="; tail -20 exp-std/logs_purestd/flux_pipeline_latest.log 2>/dev/null
echo "=== 产物 ==="; ls -la exp-std/data/shards_img_flux16/ exp-std/data/shards_std_flux16/ 2>/dev/null | head -12
