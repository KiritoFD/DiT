#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1
D="exp-std/reeval/20261003-163412-v46-std-adaln4-top10-p1.0__0025000"

for S in eval200 seen; do
  echo "===== poster: $S ====="
  PYTHONPATH=. /opt/conda/envs/cu121/bin/python -u tools/make_poster.py \
      --dir "$D" --set "$S" --cols 16 2>&1 | tail -3
done

echo
echo "===== 目录里的 csv ====="
ls -la "$D"/*.csv "$D"/posters/ 2>&1 | head -14
echo "===== summary 内容 ====="
head -4 "$D/eval_stdskel_summary.csv" 2>&1
