#!/bin/bash
# 对多个 ckpt 算完整指标（CPU）：先 batch_eval --save-samples 存图，再算 OCR/KID
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
SET="seen:assets/eval_v13_seen_fixed.csv:20"
OUT=/root/Workspace/xy/DiT/assets/metrics_cpu.csv
rm -f $OUT
for spec in "v15c_fixed_210k:assets/results/v15c_fixed/20260921-212920-v15c-multistyle-k4-ctx/checkpoints/0210000.pt" \
            "v13base_155k:assets/results/v13_base_50k/20260917-211905-v13-base-50k/checkpoints/0155000.pt" \
            "v13wd01_125k:assets/results/v13_wd01/20260918-210256-v13-base-50k/checkpoints/0125000.pt"; do
  name="${spec%%:*}"; ck="${spec#*:}"
  [ -f "$ck" ] || { echo "[skip] $name: ckpt 不存在"; continue; }
  echo "=== $name ==="
  D=/tmp/_metrics_$name; rm -rf $D; mkdir -p $D
  CUDA_VISIBLE_DEVICES="" nice -n 10 $PY -u tools/eval/eval_stdskel_batch.py \
    --results-dir $D --ckpt-override "$ck" --device cpu \
    --sets "$SET" --dit-batch 4 --vae-batch 4 --save-samples 2>&1 | grep -E "ssim=" | tail -1
  SD=$D/eval_samples_ctrl
  ls $SD 2>/dev/null | wc -l | sed "s/^/    存图: /"
  CUDA_VISIBLE_DEVICES="" nice -n 10 $PY -u tools/eval_extra_metrics.py \
    --dir $SD --csv assets/eval_v13_seen_fixed.csv --ocr rapidocr \
    --kid-feat inception --device cpu --out $OUT 2>&1 | grep -E "OCR|KID|->" | tail -4
done
echo "=== ALL DONE ==="
cat $OUT
