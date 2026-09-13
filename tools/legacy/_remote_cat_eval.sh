#!/bin/bash
# Concatenate all eval_auto_*.json across run dirs
cd /root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear
for d in */checkpoints; do
    echo "===_DSH_SEP_==="
    cat "$d/eval_auto_*.json" 2>/dev/null
    echo ""
done
