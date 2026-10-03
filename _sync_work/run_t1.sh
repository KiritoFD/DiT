#!/usr/bin/env bash
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
CSV=assets/eval_v13_strict.csv
mkdir -p assets/t1_logs
for D in assets/ink_eval/*__strict; do
  RUN=$(basename "$D" | sed 's/__strict$//')
  [ -f "$D/g0.png" ] || { echo "[skip] $RUN"; continue; }
  echo "=============== $RUN ==============="
  $PY tools/probe_style_variance.py --samples "$D" --eval-csv "$CSV" \
      --out "assets/t1_${RUN}.json" 2>&1 | tee "assets/t1_logs/${RUN}.log" | \
      grep -E "生成/GT|→ 生成|总体 F|书家 "
done
echo "ALL DONE"
