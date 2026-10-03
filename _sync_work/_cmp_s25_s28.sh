#!/bin/bash
cd /root/Workspace/xy/DiT
S25=assets/results/s25_ids_pretrain/20260831-033258-s25-ids-pretrain
echo "=== s25 early loss (first eval-window steps) ==="
grep -a -E 'step=000[12][0-9]{3}' $S25/log.txt | head -6
echo ""
echo "=== s25 eval_auto metrics (all) ==="
for f in $(ls $S25/checkpoints/eval_auto_*.json | sort); do
  /opt/conda/bin/python -c "import json;d=json.load(open('$f'));print(f\"step {d['step']:>6}  ssim={d['ssim']:.4f}  lpips={d['lpips']:.4f}  mse={d['mse']:.4f}\")"
done
