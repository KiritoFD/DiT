#!/bin/bash
# 诊断: 编码为什么崩/卡
cd /root/Workspace/xy/DiT || exit 1

echo "########## 1. 日志 ##########"
ls -lt exp-std/logs_purestd/ 2>/dev/null | head -5
echo "--- flux_pipeline_latest.log 全文(前 80 行) ---"
head -80 exp-std/logs_purestd/flux_pipeline_latest.log 2>/dev/null
echo "--- 尾 40 行 ---"
tail -40 exp-std/logs_purestd/flux_pipeline_latest.log 2>/dev/null

echo
echo "########## 2. tmux 会话 ##########"
tmux ls 2>&1 | head -5

echo
echo "########## 3. 相关进程 ##########"
ps -eo pid,ppid,etime,pcpu,pmem,rss,stat,cmd | grep -E 'encode_flux|latent_signal|flux_pipeline|train\.py' | grep -v grep | head -30
echo "--- 线程/worker 数 ---"
ps -ef | grep -c 'encode_flux_latents'

echo
echo "########## 4. GPU 上的进程 ##########"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

echo
echo "########## 5. 产物 ##########"
for d in exp-std/data/shards_img_flux16 exp-std/data/shards_std_flux16 exp-std/signal_arena_full; do
  echo "--- $d"; ls -la "$d" 2>/dev/null | head -6
done

echo
echo "########## 6. 内存/磁盘 ##########"
free -g | head -2
df -h /root | tail -1

echo
echo "########## 7. 图片数与 dmesg 里的 OOM ##########"
ls data/top10_style23/imgs/*.png 2>/dev/null | wc -l
ls data/top10_style23/std/*.png 2>/dev/null | wc -l
dmesg -T 2>/dev/null | tail -5
