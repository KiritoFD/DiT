#!/bin/bash
# 竞技场/G训练是否真在 GPU 上跑? 有没有被挤? 出结果没?
cd /root/Workspace/xy/DiT || exit 1

echo "########## 1. GPU 上的进程 (pid / 名字 / 显存) ##########"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null

echo
echo "########## 2. GPU 总览 ##########"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "########## 3. 相关进程 (PID/运行时长/CPU%/内存%/命令) ##########"
ps -eo pid,etime,pcpu,pmem,cmd | grep -E 'latent_signal_arena|initmix_sweep|train\.py' | grep -v grep

echo
echo "########## 4. 竞技场产物 ##########"
ls -la exp-std/signal_arena/ 2>/dev/null | head -14

echo
echo "########## 5. 已落的 csv ##########"
for f in exp-std/signal_arena/margin.csv exp-std/signal_arena/inversion.csv; do
  if [ -f "$f" ]; then echo "--- $f"; cat "$f"; fi
done

echo
echo "########## 6. 续训进度 ##########"
ls -t exp-std/runs_purestd/*p1.0/checkpoints/*.pt 2>/dev/null | head -1
tail -2 exp-std/logs_purestd/resume_*.log 2>/dev/null | tail -3
