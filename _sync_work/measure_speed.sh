#!/usr/bin/env bash
# 稳态性能归因: GPU 到底在等谁? (功率/利用率/步速/CPU/线程/数据加载)
set -u
cd /root/Workspace/xy/DiT || exit 1

echo "########## [1] GPU 型号 + 稳态功耗 ##########"
nvidia-smi --query-gpu=name,power.draw,power.limit,power.default_limit,utilization.gpu,utilization.memory,memory.used,temperature.gpu,clocks.sm,clocks.max.sm --format=csv

echo
echo "########## [2] 进程 ##########"
ps -eo pid,ppid,pcpu,pmem,etime,stat,cmd --sort=-pcpu | grep -E 'train\.py|launch_AB' | grep -v grep | head -6

echo
echo "########## [3] GPU 上跑的是谁 ##########"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv

echo
echo "########## [4] A 路日志的步速轨迹 (最近 12 条) ##########"
LOG=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
echo "log=$LOG"
if [ -n "${LOG:-}" ]; then
  grep -aE 'Steps/Sec|Step/Sec|steps/s' "$LOG" | tail -12
  echo "--- 日志最后 6 行 ---"
  tail -6 "$LOG"
fi

echo
echo "########## [5] 线程级 CPU (看 dataloader worker 是否在抢) ##########"
PID=$(pgrep -f 'src/train/train.py' | head -1)
if [ -n "${PID:-}" ]; then
  echo "pid=$PID  线程数: $(ls /proc/$PID/task | wc -l)"
  top -b -n 1 -H -p "$PID" 2>/dev/null | tail -18
  echo "--- 该进程族的全部线程 TOP ---"
  ps -eLo pid,tid,pcpu,comm --sort=-pcpu | head -12
else
  echo "(没有 train.py 在跑?)"
fi

echo
echo "########## [6] CPU 总体 + 核数 ##########"
nproc
uptime
