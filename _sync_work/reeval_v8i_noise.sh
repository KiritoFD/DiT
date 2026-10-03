#!/bin/bash
# reeval_v8i_noise.sh — v8i best ckpt 重评 (v8 协议落盘图) + metrics_png 噪点
# 注意: v8e 训练占 GPU, eval 采样短暂分时; 用较低的 DIT batch 减小干扰
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
PY=/opt/conda/envs/cu121/bin/python
OUT=assets/results/v8_3stage/reeval_v8i_noise
mkdir -p "$OUT"
CSV=assets/eval_fame_strict_clean_v8.csv
SKLAT=data/skel/final_skel_latents_fame_1px_v8
SKROOT=data/skel/final_skel1_fame_v8
IMGROOT=data/imgs/final_imgs_fame_v8
A=assets/results/v8_3stage/A_main_final.pt
V8I=assets/results/v8_3stage/v8i/20260903-194339-v8i-unfreeze-repa-early/checkpoints/0030000.pt

# 重评: v8i 是完整模型 (unfreeze 训练, ckpt 含 main+ctrl)
$PY src/eval/eval_ctrl_ckpt.py \
  --main-ckpt "$A" --ctrl-ckpt "$V8I" \
  --eval-csv "$CSV" --skel-latent-dir "$SKLAT" --skel-root "$SKROOT" --img-root "$IMGROOT" \
  --out-dir "$OUT/v8i_030000" --n 100 --cfg 0.7 --steps 50 --dit-batch 4 \
  > "$OUT/v8i_030000.log" 2>&1
echo "[repa] rc=$? skip"
# 噪点: ctrl 图 + base 图 (v8i 解冻后 base 也值得测)
$PY src/eval/metrics_png.py --dir "$OUT/v8i_030000/ctrl" --tag ctrl --n 100 --out "$OUT/v8i_030000/ctrl_noise.json" > "$OUT/v8i_030000/noise_ctrl.log" 2>&1
echo "[noise] ctrl rc=$?"
[ -d "$OUT/v8i_030000/base" ] && $PY src/eval/metrics_png.py --dir "$OUT/v8i_030000/base" --tag base --n 100 --out "$OUT/v8i_030000/base_noise.json" > "$OUT/v8i_030000/noise_base.log" 2>&1 && echo "[noise] base rc=$?"
echo ALL_DONE