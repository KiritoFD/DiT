#!/bin/bash
cd /root/Workspace/xy/DiT
for f in assets/results/v13_12ch_post/eval_stdskel_summary.csv \
         assets/results/v13_base_50k/eval_stdskel_summary.csv \
         assets/results/v25_stdskel/eval_stdskel_summary.csv \
         assets/results/v10b_stdskel_fame3_c41x_cos_e/eval_stdskel_summary.csv; do
  [ -f "$f" ] || continue
  echo "=== $f"
  head -1 "$f"
  awk -F, 'NR>1 {printf "  step=%-8s set=%-8s n=%-4s ssim=%-8s mse=%-8s lpips=%s\n", $2,$3,$4,$5,$11,$12}' "$f" | head -8
done
