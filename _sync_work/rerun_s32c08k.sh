#!/bin/bash
# rerun_s32c08k.sh — s32c 0080000.pt 同口径重评 (v8 资产协议) + v8b/v8c/s32c 的 ctrl 图跑 metrics_png 噪点指标
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
PY=/opt/conda/envs/cu121/bin/python
OUT=assets/results/v8_3stage/reeval_same_protocol
CSV=assets/eval_fame_strict_clean_v8.csv
SKLAT=data/skel/final_skel_latents_fame_1px_v8
SKROOT=data/skel/final_skel1_fame_v8
IMGROOT=data/imgs/final_imgs_fame_v8
A=assets/results/v8_3stage/A_main_final.pt
S32C=assets/results/s32c_chain/20260902-004653-s32c-repa-longconv/checkpoints/0080000.pt

# 1) s32c 0080000 重评 (main=c自身, ctrl=c自身)
$PY src/eval/eval_ctrl_ckpt.py \
  --main-ckpt "$S32C" --ctrl-ckpt "$S32C" \
  --eval-csv "$CSV" --skel-latent-dir "$SKLAT" --skel-root "$SKROOT" --img-root "$IMGROOT" \
  --out-dir "$OUT/s32c_repa80k" --n 100 --cfg 0.7 --steps 50 \
  > "$OUT/s32c_repa80k.log" 2>&1
echo "[s32c80k] rc=$? ctrl=$(cat $OUT/s32c_repa80k/metrics.json 2>/dev/null | grep -o '"ssim_mean": [0-9.]*' | head -2 | tr '\n' ' ')"

# 2) 噪点指标: 对每个重评输出的 ctrl 图跑 metrics_png
for tag in v8a_base v8b_ctrl v8c_repa s32c_repa80k; do
  DIR="$OUT/$tag/ctrl"
  [ -d "$DIR" ] || { echo "[$tag] no ctrl dir, skip"; continue; }
  $PY src/eval/metrics_png.py --dir "$DIR" --tag ctrl --n 100 --out "$OUT/$tag/ctrl_noise_metrics.json" > "$OUT/$tag/noise.log" 2>&1
  echo "[$tag] noise rc=$?"
done
echo ALL_DONE