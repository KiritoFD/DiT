#!/bin/bash
# rerun_ctrl_reeval.sh — 同口径重评 (v8 资产协议): v8b best / v8c best / s32c best / v8a base
# eval_ctrl_ckpt.py: --main-ckpt --ctrl-ckpt --eval-csv --skel-latent-dir --skel-root --img-root --out-dir --n --cfg --steps
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
PY=/opt/conda/envs/cu121/bin/python
OUT=assets/results/v8_3stage/reeval_same_protocol
mkdir -p "$OUT"
CSV=assets/eval_fame_strict_clean_v8.csv
SKLAT=data/skel/final_skel_latents_fame_1px_v8
SKROOT=data/skel/final_skel1_fame_v8
IMGROOT=data/imgs/final_imgs_fame_v8

A=assets/results/v8_3stage/A_main_final.pt
V8B=assets/results/v8_3stage/B_ctrl_best.pt
V8C=assets/results/v8_3stage/v8c/20260903-024429-v8c-s32-repa/checkpoints/0015000.pt
S32C=assets/results/s32c_chain/20260902-004653-s32c-repa-longconv/checkpoints/0040000.pt

run() {
  $PY src/eval/eval_ctrl_ckpt.py \
    --main-ckpt "$2" --ctrl-ckpt "$3" \
    --eval-csv "$CSV" --skel-latent-dir "$SKLAT" --skel-root "$SKROOT" --img-root "$IMGROOT" \
    --out-dir "$OUT/$1" --n 100 --cfg 0.7 --steps 50 \
    > "$OUT/$1.log" 2>&1
  echo "[$1] rc=$? -> $(cat $OUT/$1/metrics.json 2>/dev/null | head -c 400)"
}

echo "=== v8a-base (main only) ==="
run v8a_base "$A" ""

echo "=== v8b-ctrl ==="
run v8b_ctrl "$A" "$V8B"

echo "=== v8c-repa ==="
run v8c_repa "$A" "$V8C"

echo "=== s32c-repa (旧链, v8协议重评) ==="
run s32c_repa "$S32C" "$S32C"

echo ALL_DONE