#!/bin/bash
cd /root/Workspace/xy/DiT/assets/results/s28_std_dino_pretrain
LATEST=$(ls -dt 2026*/ | head -1)
echo "run: $LATEST"
echo "=== s28 (PCA+OT) eval ==="
for f in $(ls "$LATEST"checkpoints/eval_auto_*.json 2>/dev/null | sort); do
  /opt/conda/bin/python -c "import json;d=json.load(open('$f'));print(f\"  step {d['step']:>5}  ssim={d['ssim']:.4f}  lpips={d['lpips']:.4f}  mse={d['mse']:.4f}\")"
done
echo ""
echo "=== s21 (真迹 DINO baseline) reference ==="
echo "  step 1000  ssim=0.4204  lpips=0.5128"
echo "  step 2000  ssim=0.4393  lpips=0.4801"
echo "  step 3000  ssim=0.4398  lpips=0.4724"
echo "  step 27500 ssim=0.4664  lpips=0.4430 (best)"
