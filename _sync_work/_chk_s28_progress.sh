#!/bin/bash
cd /root/Workspace/xy/DiT/assets/results/s28_std_dino_pretrain
L=$(ls -dt 2026*/ | head -1)
echo "run: $L"
echo "=== last steps ==="
grep -a -o 'step=[0-9]*  Total: [0-9.]*' "$L"log.txt | tail -3
echo "=== Steps/Sec ==="
grep -a -o 'Steps/Sec: [0-9.]*' "$L"log.txt | tail -2
echo "=== eval so far ==="
ls "$L"checkpoints/eval_auto_*.json 2>/dev/null | wc -l
echo "=== latest eval ==="
f=$(ls "$L"checkpoints/eval_auto_*.json 2>/dev/null | sort | tail -1)
if [ -n "$f" ]; then /opt/conda/bin/python -c "import json;d=json.load(open('$f'));print(f\"step {d['step']} ssim={d['ssim']:.4f} lpips={d['lpips']:.4f}\")"; fi
echo "=== train proc ==="
pgrep -af 'train.py.*s28' | head -1
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
