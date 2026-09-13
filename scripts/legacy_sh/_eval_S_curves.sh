#!/bin/bash
# 对 S 的所有保留 ckpt 用 fast100（固定 100 张）跑自由采样 eval
cd /root/Workspace/xy/DiT
mkdir -p gen_curves_S
for ckpt in results/exp_s_scratch/20260814-170506-DiT-3Cond-S-2/checkpoints/0*.pt; do
  step=$(basename "$ckpt" .pt)
  out="gen_curves_S/$step"
  if [ -f "$out.json" ]; then echo "skip $step (exists)"; continue; fi
  echo "=== eval S step=$step ==="
  /opt/conda/bin/python eval_gen.py --ckpt "$ckpt" --model DiT-3Cond-S/2 --use-lora 0 \
    --pretrained null --csv fast100.csv --n 100 --steps 50 --cfg 4.0 --batch 16 --out "$out"
done
echo "ALL_S_DONE"
