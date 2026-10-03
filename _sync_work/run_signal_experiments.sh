#!/bin/bash
# run_signal_experiments.sh — GPU 空闲期一次性做齐信号实验
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
PY=/opt/conda/envs/cu121/bin/python
OUT=/tmp/signal_exp
mkdir -p $OUT

echo "===== [1/3] v8e best (22.5k, ctrl SOTA 0.7761) 同协议重评 ====="
A=assets/results/v8_3stage/A_main_final.pt
V8E=assets/results/v8_3stage/v8e/20260904-140950-v8e-repa-strong/checkpoints/0022500.pt
$PY src/eval/eval_ctrl_ckpt.py \
  --main-ckpt "$A" --ctrl-ckpt "$V8E" \
  --eval-csv assets/eval_fame_strict_clean_v8.csv \
  --skel-latent-dir data/skel/final_skel_latents_fame_1px_v8 \
  --skel-root data/skel/final_skel1_fame_v8 --img-root data/imgs/final_imgs_fame_v8 \
  --out-dir "$OUT/v8e_022500" --n 100 --cfg 0.7 --steps 50 --dit-batch 4 \
  > "$OUT/v8e_022500.log" 2>&1
echo "[v8e] rc=$?"
[ -f "$OUT/v8e_022500/metrics.json" ] && python3 -c "import json; j=json.load(open('$OUT/v8e_022500/metrics.json')); print('ctrl.ssim=%.4f base.ssim=%.4f lpips=%.4f mse=%.4f' % (j['ctrl']['ssim_mean'], j['base']['ssim_mean'], j['ctrl']['lpips_mean'], j['ctrl']['mse_mean']))"

echo "===== [2/3] v8f/v8g 没有早停的问题 — 修 config 后不跑 (等用户确认) ====="
echo "skip"

echo "===== [3/3] DINO CLS 全量检索实验 (同字样本级/原型级/跨书家) ====="
$PY /tmp/dino_cls_aggregate.py 2>&1 | tail -12
echo ALL_DONE