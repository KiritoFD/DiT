#!/bin/bash
cd /root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear
for d in */checkpoints; do
  cat "$d"/eval_auto_*.json 2>/dev/null
  echo ""
done
