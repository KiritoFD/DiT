#!/bin/bash
# eval_all_s31.sh — 对 s31 全部 ckpt 用正确 skel latent 逐批 eval (GPU), 汇总 summary
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT

CKDIR=$(ls -td assets/results/s31_ctrl_gt_skel_1px/*/ 2>/dev/null | head -1)/checkpoints
MAIN=assets/results/s30_dino_char_strong_pretrain/20260901-052520-s30-dino-char-strong-pretrain/checkpoints/0132500.pt
EVAL=assets/eval_fame_strict_clean.csv
OUTROOT="$CKDIR/manual_eval_correct_skel"
SUMMARY="$OUTROOT/summary.json"

echo "ckdir=$CKDIR"
mkdir -p "$OUTROOT"
echo "[" > "$SUMMARY"

FIRST=1
for f in $(ls "$CKDIR"/*.pt 2>/dev/null | grep -v manual_eval | sort); do
  STEP=$(basename "$f" .pt)
  OUT="$OUTROOT/step$STEP"
  if [ -f "$OUT/metrics.json" ]; then
    echo "[skip] step $STEP (already done)"
  else
    echo "[eval] step $STEP ..."
    timeout 1200 /opt/conda/envs/cu121/bin/python src/eval/eval_ctrl_ckpt.py \
        --main-ckpt "$MAIN" \
        --ctrl-ckpt "$f" \
        --eval-csv "$EVAL" \
        --skel-latent-dir data/skel/final_skel_latents_fame_1px \
        --img-root data/imgs/final_imgs_256 --skel-root data/skel/final_skel1_fame \
        --out-dir "$OUT" \
        --n 100 --cfg 0.7 --steps 50 --device cuda \
        --dit-batch 100 --vae-batch 32 \
        > /tmp/eval_s31_step${STEP}.log 2>&1
    RC=$?
    echo "  -> rc=$RC"
    if [ $RC -ne 0 ]; then
      tail -20 /tmp/eval_s31_step${STEP}.log
    fi
  fi
  if [ -f "$OUT/metrics.json" ]; then
    if [ $FIRST -eq 0 ]; then echo "," >> "$SUMMARY"; fi
    /opt/conda/envs/cu121/bin/python - <<EOF
import json
j = json.load(open("$OUT/metrics.json"))
row = {
  "step": int("$STEP"),
  "ctrl_ssim": round(j["ctrl"].get("ssim_mean", 0), 4),
  "ctrl_mse": round(j["ctrl"].get("mse_mean", 0), 5),
  "ctrl_skel_iou": round(j["ctrl"].get("skel_iou_mean", 0), 4),
  "ctrl_lpips": round(j["ctrl"].get("lpips_mean", 0), 4),
  "base_ssim": round(j["base"].get("ssim_mean", 0), 4),
  "delta_ssim": round(j.get("delta_ssim", 0), 4),
  "delta_mse": round(j.get("delta_mse", 0), 5),
}
print(json.dumps(row, ensure_ascii=False))
EOF
    FIRST=0
  fi
done
echo "]" >> "$SUMMARY"
echo
echo "======== 汇总完成: $SUMMARY ========"
cat "$SUMMARY"