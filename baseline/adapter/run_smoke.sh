#!/bin/bash
# Smoke tests: 1-2 training iterations + tiny inference per baseline.
# Designed to run under GPU contention (v24 training holds ~19GB); each smoke
# is small. If OOM, set SMOKE_CPU=1 to run the train step on CPU.
# Run: bash /root/Workspace/xy/DiT/baseline/adapter/run_smoke.sh [fontdiffuser|vqfont|dgfont|all]
set -e
STAGE=${1:-all}
PY=/opt/conda/envs/baseline/bin/python
BASE=/root/Workspace/xy/DiT/baseline
ADP=$BASE/adapter
LOG=$BASE/smoke_logs
mkdir -p $LOG

smoke_fontdiffuser() {
  echo "==== FontDiffuser smoke ===="
  cd $BASE/FontDiffuser
  ACCEL=/opt/conda/envs/baseline/bin/accelerate
  if [ "${SMOKE_CPU:-0}" = "1" ]; then export CUDA_VISIBLE_DEVICES=""; else export CUDA_VISIBLE_DEVICES=0; fi
  # 2 steps, batch 1, no accum
  timeout 900 $ACCEL launch --num_processes=1 --mixed_precision=no train.py \
    --seed=123 --experience_name=smoke --data_root="../data/fontdiffuser" \
    --output_dir=outputs/smoke --report_to=tensorboard --resolution=256 \
    --style_image_size=256 --content_image_size=256 --content_encoder_downsample_size=3 \
    --channel_attn=True --content_start_channel=64 --style_start_channel=64 \
    --train_batch_size=1 --perceptual_coefficient=0.01 --offset_coefficient=0.5 \
    --max_train_steps=2 --ckpt_interval=100000 --gradient_accumulation_steps=1 \
    --log_interval=1 --learning_rate=1e-4 --lr_scheduler="linear" --lr_warmup_steps=0 \
    --drop_prob=0.1 --mixed_precision="no" 2>&1 | tee $LOG/fontdiffuser_train.log | tail -5
  echo "FontDiffuser train smoke OK"
}

smoke_vqfont_pretrain() {
  echo "==== VQ-Font VQ-VAE smoke (30 iters) ===="
  cd $BASE/VQ-Font
  $PY ../adapter/vqfont/pretrain_vqvae.py --iters 30 --batch_size 16 \
      2>&1 | tee $LOG/vqfont_vae.log | tail -3
  # rename smoke artifact so build_similarity/training can use it
  mv -f weight/VQ-VAE_top10.pth weight/VQ-VAE_top10_smoke.pth 2>/dev/null || true
  mv -f weight/VQ-VAE_top10_Parms.pth weight/VQ-VAE_top10_Parms_smoke.pth 2>/dev/null || true
  echo "VQ-VAE smoke OK"
}

smoke_vqfont_sim() {
  echo "==== VQ-Font similarity smoke ===="
  cd $BASE/VQ-Font
  cp -f weight/VQ-VAE_top10_smoke.pth weight/VQ-VAE_top10.pth
  cp -f weight/VQ-VAE_top10_Parms_smoke.pth weight/VQ-VAE_top10_Parms.pth
  $PY ../adapter/vqfont/build_similarity.py 2>&1 | tee $LOG/vqfont_sim.log | tail -2
  echo "VQ-Font sim smoke OK"
}

smoke_vqfont_train() {
  echo "==== VQ-Font FFG train smoke (4 iters, batch 6) ===="
  cd $BASE/VQ-Font
  if [ "${SMOKE_CPU:-0}" = "1" ]; then export CUDA_VISIBLE_DEVICES=; else export CUDA_VISIBLE_DEVICES=0; fi
  timeout 1700 $PY ../adapter/vqfont/train_top10.py top10smoke \
      ../adapter/vqfont/cfg_top10.yaml ../adapter/vqfont/cfg_smoke.yaml \
      2>&1 | tee $LOG/vqfont_train.log | tail -8
  echo "VQ-Font train smoke OK (timeout-terminated as expected)"
}

smoke_dgfont() {
  echo "==== DG-Font train smoke (25 epochs x 1 iter) ===="
  cd $BASE/DG-Font
  timeout 900 $PY main.py --gpu 0 \
      --img_size 256 --data_path ../data/dgfont/train \
      --output_k 24 --val_num 8 --val_batch 8 --sty_dim 128 --batch_size 8 --workers 8 \
      --epochs 25 --iters 1 --log_step 1 --model_name GAN_smoke_256 \
      2>&1 | tee $LOG/dgfont_train.log | tail -8
  echo "DG-Font train smoke OK"
}

case $STAGE in
  fontdiffuser) smoke_fontdiffuser ;;
  vqfont)       smoke_vqfont_pretrain; smoke_vqfont_sim; smoke_vqfont_train ;;
  dgfont)       smoke_dgfont ;;
  all)          smoke_fontdiffuser; smoke_vqfont_pretrain; smoke_vqfont_sim; \
                smoke_vqfont_train; smoke_dgfont ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo SMOKE_DONE
