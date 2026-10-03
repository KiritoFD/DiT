#!/bin/bash
# eval_s32_cfg07.sh — 用 cfg=0.7 对 s32(弱REPA) 已有 ckpt 做正确 eval (含 main override)
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
MAIN=assets/results/s30_dino_char_strong_pretrain/20260901-052520-s30-dino-char-strong-pretrain/checkpoints/0132500.pt
EVAL=assets/eval_fame_strict_clean.csv
S32=$(ls -dt assets/results/s32_repa_finetune/*/ 2>/dev/null | head -1)
CKDIR="$S32/checkpoints"
OUT="$CKDIR/manual_eval_cfg07"

echo "S32=$S32"
echo "CKPT dir=$CKDIR"
ls "$CKDIR"/*.pt 2>/dev/null | grep -v manual_eval | grep -v ext_metrics
mkdir -p "$OUT"

for f in $(ls "$CKDIR"/*.pt 2>/dev/null | grep -v manual_eval | sort); do
  STEP=$(basename "$f" .pt)
  D="$OUT/step$STEP"
  if [ -f "$D/metrics.json" ]; then
    echo "[skip] $STEP (done)"
    continue
  fi
  echo "[eval] s32 weak REPA step $STEP (cfg=0.7) ..."
  timeout 1500 $PY src/eval/eval_ctrl_ckpt.py \
      --main-ckpt "$MAIN" \
      --ctrl-ckpt "$f" \
      --eval-csv "$EVAL" \
      --skel-latent-dir data/skel/final_skel_latents_fame_1px \
      --img-root data/imgs/final_imgs_256 --skel-root data/skel/final_skel1_fame \
      --out-dir "$D" \
      --n 100 --cfg 0.7 --steps 50 --device cuda \
      --dit-batch 100 --vae-batch 32 \
      > /tmp/eval_s32_cfg07_step${STEP}.log 2>&1
  RC=$?
  echo "  rc=$RC"
  [ $RC -ne 0 ] && tail -15 /tmp/eval_s32_cfg07_step${STEP}.log
done

# 汇总
$PY - <<'EOF'
import json, glob, os
S32 = "/root/Workspace/xy/DiT/assets/results/s32_repa_finetune/20260901-183011-s32-repa-finetune"
rows = []
for m in sorted(glob.glob(os.path.join(S32, "checkpoints/manual_eval_cfg07/step*/metrics.json"))):
    step = int(os.path.basename(os.path.dirname(m)).replace("step", ""))
    j = json.load(open(m))
    rows.append({"step": step,
        "ctrl_ssim": j["ctrl"]["ssim_mean"], "ctrl_mse": j["ctrl"]["mse_mean"],
        "ctrl_skel_iou": j["ctrl"]["skel_iou_mean"],
        "delta_ssim": j.get("delta_ssim", 0),
        "ctrl_lpips": j["ctrl"].get("lpips_mean", 0)})
rows.sort(key=lambda r: r["step"])
out = os.path.join(S32, "checkpoints/manual_eval_cfg07/summary.json")
json.dump(rows, open(out, "w"), indent=1, ensure_ascii=False)
if not rows:
    print("(no rows)"); raise SystemExit
print(f"{'step':>7} | {'ctrl.ssim':>9} {'ctrl.mse':>9} {'skel_iou':>8} {'lpips':>7} | {'dSSIM':>7}")
for r in rows:
    print(f"{r['step']:>7} | {r['ctrl_ssim']:>9.4f} {r['ctrl_mse']:>9.4f} {r['ctrl_skel_iou']:>8.4f} {r['ctrl_lpips']:>7.4f} | {r['delta_ssim']:>+7.4f}")
print(f"\n-> {out}")
EOF