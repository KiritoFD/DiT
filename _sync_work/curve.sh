#!/usr/bin/env bash
# 取 A 路训练曲线 (从日志 + eval_auto_*.json, 绕开 summary 被重置的 bug)
set -u
cd /root/Workspace/xy/DiT || exit 1
LOG=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
echo "log=$LOG"

echo
echo "=== [1] 每次 eval 的汇总行 ==="
grep -a 'in-mem-eval] step ' "$LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^\[[^]]*\] //'

echo
echo "=== [2] 每次 eval 的耗时 ==="
grep -a 'done in' "$LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^\[[^]]*\] //'

echo
echo "=== [3] eval_auto json 曲线 ==="
D=$(sed -n '1p' exp-std/runs_AB/_active_ckpt_dir.txt 2>/dev/null | xargs dirname)
echo "run dir=$D"
for f in $D/eval_auto_*.json; do
  [ -f "$f" ] || continue
  echo -n "  $(basename $f): "
  tr -d '\n ' < "$f" | head -c 300
  echo
done

echo
echo "=== [4] 当前进度 ==="
grep -a 'Steps/Sec' "$LOG" 2>/dev/null | tail -1 | sed 's/\x1b\[[0-9;]*m//g'
date
