#!/bin/bash
cd /root/Workspace/xy/DiT
for d in _calib_s1_v32 _calib_w7raw_v32; do
  f=assets/results/$d/eval_stdskel_summary.csv
  echo "=== $d"
  head -1 "$f"
  grep -E ',(seen|strict),' "$f" | head -4
done
