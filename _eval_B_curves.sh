#!/bin/bash
# 对 B（XL-head 无 LoRA）的所有保留 ckpt 用 fast100 跑自由采样 eval
cd /root/Workspace/xy/DiT
mkdir -p gen_curves_B
for ckpt in results/exp_xl_head/20260814-184935-DiT-3Cond-XL-2/checkpoints/0*.pt; do
  step=$(basename "$ckpt" .pt)
  out="gen_curves_B/$step"
  if [ -f "$out.json" ]; then echo "skip $step (exists)"; continue; fi
  echo "=== eval B step=$step ==="
  /opt/conda/bin/python eval_gen.py --ckpt "$ckpt" --model DiT-3Cond-XL/2 --use-lora 0 \
    --pretrained pretrained_models/DiT-XL-2-256x256.pt --csv fast100.csv --n 100 \
    --steps 50 --cfg 4.0 --batch 8 --out "$out"
done
echo "ALL_B_DONE"
