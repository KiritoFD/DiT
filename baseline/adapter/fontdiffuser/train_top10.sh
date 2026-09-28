#!/bin/bash
# FontDiffuser @ top10_style23, adapted to 256x256 (dataset resolution).
# Faithful recipe otherwise: UNet(64,128,256,512) + MCA + StyleRSI, DDPM scaled_linear,
# linear LR schedule w/ 10k warmup, drop_prob 0.1, perceptual 0.01, offset 0.5.
# Phase 1 only (phase 2 needs SCR pretraining which the repo has not released).
# NOTE: run inside baseline conda env from /root/Workspace/xy/DiT/baseline/FontDiffuser
set -e
ENV=${ENV:-/opt/conda/envs/baseline/bin/python}
STEPS=${STEPS:-150000}
BS=${BS:-4}
ACC=${ACC:-4}          # effective batch = BS*ACC = 16 (paper setting)
OUT=${OUT:-outputs/top10_phase1_256}

accelerate launch --num_processes=1 --mixed_precision=no train.py \
    --seed=123 \
    --experience_name="fontdiffuser_top10_phase1" \
    --data_root="../data/fontdiffuser" \
    --output_dir="$OUT" \
    --report_to="tensorboard" \
    --resolution=256 \
    --style_image_size=256 \
    --content_image_size=256 \
    --content_encoder_downsample_size=3 \
    --channel_attn=True \
    --content_start_channel=64 \
    --style_start_channel=64 \
    --train_batch_size=$BS \
    --perceptual_coefficient=0.01 \
    --offset_coefficient=0.5 \
    --max_train_steps=$STEPS \
    --ckpt_interval=10000 \
    --gradient_accumulation_steps=$ACC \
    --log_interval=50 \
    --learning_rate=1e-4 \
    --lr_scheduler="linear" \
    --lr_warmup_steps=10000 \
    --drop_prob=0.1 \
    --mixed_precision="no"
