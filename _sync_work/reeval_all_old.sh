#!/bin/bash
# reeval_all_old.sh — 固定 v8 eval 协议重评所有历史 ckpt
# 协议: eval_fame_strict_clean_v8.csv + data/skel/final_skel_latents_fame_1px_v8 + data/imgs/final_imgs_fame_v8, cfg 0.7, n=100, 50步
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
PY=/opt/conda/envs/cu121/bin/python
OUT=assets/results/v8_3stage/reeval_fixed_protocol
mkdir -p "$OUT"
CSV=assets/eval_fame_strict_clean_v8.csv
SKLAT=data/skel/final_skel_latents_fame_1px_v8
SKROOT=data/skel/final_skel1_fame_v8
IMGROOT=data/imgs/final_imgs_fame_v8
R=sassets  # placeholder

# 固定路径基准
A=assets/results/v8_3stage/A_main_final.pt

# 历史 base ckpt (只有 ema)
S21=assets/results/s21_fame_flow_v2/20260829-232329-s21-fame-flow-v2/checkpoints/0040000.pt
S30=assets/results/s30_dino_char_strong_pretrain/20260901-052520-s30-dino-char-strong-pretrain/checkpoints/0132500.pt
# 历史 ctrl ckpt (只有 ctrl/ema; main 用 S30)
S31=assets/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1px/checkpoints/0042500.pt
# 历史 repa ckpt (model+ctrl 都有)
S32=assets/results/s32_repa_finetune/20260901-183011-s32-repa-finetune/checkpoints/0012500.pt
S32B=assets/results/s32b_repa_strong/20260901-204250-s32b-repa-strong/checkpoints/0015000.pt
S32C=assets/results/s32c_chain/20260902-004653-s32c-repa-longconv/checkpoints/0080000.pt
S32D=assets/results/s32c_chain/20260902-093208-s32d-repa-super/checkpoints/0010000.pt

run() { # name main ctrl
  local name=$1 main=$2 ctrl=$3
  $PY src/eval/eval_ctrl_ckpt.py \
    --main-ckpt "$main" --ctrl-ckpt "$ctrl" \
    --eval-csv "$CSV" --skel-latent-dir "$SKLAT" --skel-root "$SKROOT" --img-root "$IMGROOT" \
    --out-dir "$OUT/$name" --n 100 --cfg 0.7 --steps 50 \
    > "$OUT/$name.log" 2>&1
  local rc=$?
  local ssim="-"
  [ -f "$OUT/$name/metrics.json" ] && ssim=$(/opt/conda/bin/python -c "import json; j=json.load(open('$OUT/$name/metrics.json')); print('ctrl.ssim=%.4f base.ssim=%.4f' % (j.get('ctrl',{}).get('ssim_mean',-1), j.get('base',{}).get('ssim_mean',-1)))" 2>/dev/null)
  echo "[$name] rc=$rc $ssim"
}

echo "===== base 类 (main only) ====="
run s21_base    "$S21" ""
run s30_base    "$S30" ""

echo "===== ctrl 类 (main=S30) ====="
run s31_ctrl    "$S30" "$S31"

echo "===== repa 类 (自带 model+ctrl) ====="
run s32_repa    "$S32" "$S32"
run s32b_repa   "$S32B" "$S32B"
run s32c_repa   "$S32C" "$S32C"
run s32d_repa   "$S32D" "$S32D"

echo ALL_DONE