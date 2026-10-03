#!/usr/bin/env bash
# 快速看 A 路当前状态 (不等待)
set -u
cd /root/Workspace/xy/DiT || exit 1
date
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,clocks.current.sm --format=csv,noheader
echo
echo "=== A 路日志文件 ==="
LOG=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
echo "log=$LOG   大小=$(stat -c%s "$LOG" 2>/dev/null) 字节"
echo
echo "=== 关键里程碑 (带时间) ==="
grep -aE 'preload|Begin epoch|Reached|step=0000|alloc|Traceback|Error|Done' "$LOG" 2>/dev/null | tail -12
echo
echo "=== Steps/Sec (若有) ==="
grep -aE 'Steps/Sec|Step/Sec' "$LOG" 2>/dev/null | tail -5 || echo "(还没有步速行 -> 仍在编译/warmup)"
echo
echo "=== 最后 8 行原文 ==="
tail -8 "$LOG" 2>/dev/null
echo
echo "=== 该 run 目录 ==="
ls -t exp-std/runs_AB/ 2>/dev/null | head -3
echo
echo "=== 进程 & 线程状态 (llvm=编译中) ==="
PID=$(pgrep -f 'config src/train/configs/v50_A' | head -1)
echo "pid=$PID"
if [ -n "${PID:-}" ]; then
  ps -eLo pid,tid,pcpu,comm --sort=-pcpu | grep -E '^\s*'"$PID" | head -8
fi
