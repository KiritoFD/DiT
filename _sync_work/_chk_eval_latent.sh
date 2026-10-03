#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== eval clean 图 ==="
grep -c 'clean' assets/eval_fame_strict_clean.csv 2>/dev/null
cut -d, -f1 assets/eval_fame_strict_clean.csv | tail -n +2 | sed 's|/[^/]*$||' | sort | uniq -c
echo ""
echo "=== 所有 latent 相关目录 ==="
find . -maxdepth 2 -type d -name '*latent*' 2>/dev/null | head -20
echo ""
echo "=== eval latent 目录 ==="
for d in data/latents/final_latents_eval data/latents/final_latents_fame_eval eval_latents assets/results/s25_ids_pretrain/*/checkpoints/*latent*; do
  if [ -d "$d" ]; then echo "FOUND: $d"; ls "$d" | head -3; fi
done
echo "=== in_process_eval 如何加载 eval latent/图 ==="
grep -rn 'eval_img_root\|final_imgs\|prepare_eval_cache\|latent' src/eval/in_process_eval.py 2>/dev/null | head -15
echo "=== eval csv 前几行 ==="
head -3 assets/eval_fame_strict_clean.csv
