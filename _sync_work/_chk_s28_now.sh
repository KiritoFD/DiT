#!/bin/bash
cd /root/Workspace/xy/DiT/assets/results/s28_std_dino_pretrain
L=$(ls -dt 2026*/ | head -1)
echo "run: $L"
echo "=== checkpoints ==="
ls "$L"checkpoints/*.pt 2>/dev/null | tail -5
echo "=== eval_auto ==="
ls "$L"checkpoints/eval_auto_*.json 2>/dev/null | wc -l
echo "=== latest step in log ==="
grep -a -o 'step=[0-9]*' "$L"log.txt 2>/dev/null | tail -1
echo "=== last Total ==="
grep -a -o 'Total: [0-9.]*' "$L"log.txt 2>/dev/null | tail -1
echo "=== latest eval (if any) ==="
f=$(ls "$L"checkpoints/eval_auto_*.json 2>/dev/null | sort | tail -1)
if [ -n "$f" ]; then /opt/conda/bin/python -c "import json;d=json.load(open('$f'));print(f\"step {d['step']} ssim={d['ssim']:.4f} lpips={d['lpips']:.4f}\")"; else echo "(no eval yet)"; fi
echo "=== train proc ==="
pgrep -af 'train.py.*s28' | head -1 || echo "(s28 not running)"
