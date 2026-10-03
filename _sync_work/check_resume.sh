#!/usr/bin/env bash
# 核实: A 路真的 resume 了吗? (还是又从头跑了)
set -u
cd /root/Workspace/xy/DiT || exit 1
date
echo
echo "=== [1] launcher 日志 ==="
tail -25 exp-std/logs_AB/launch_r3.log 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g'
echo
echo "=== [2] launcher 里的 resume 行 ==="
grep -a 'resume' exp-std/logs_AB/launch_r3.log 2>/dev/null | head -5 || echo "  (没有 resume 行!)"
echo
echo "=== [3] A 日志里的恢复证据 ==="
A=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
echo "log=$A"
grep -anE 'resume|恢复|restored|Begin epoch|materialize|step=|LR\] ' "$A" 2>/dev/null | head -20
echo
echo "=== [4] 当前进度 (step 号能看出是 15k 起还是 0 起) ==="
grep -a 'Steps/Sec' "$A" 2>/dev/null | tail -3 | sed 's/\x1b\[[0-9;]*m//g'
echo
echo "=== [5] GPU + 进程 ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
pgrep -af 'train.py' | head -2 | cut -c1-220
