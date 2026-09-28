#!/bin/bash
# FontDiffuser-Latent @ top10_style23: whole model in the project latent space
# (sd-vae-ft-ema f8, 4x32x32). Data = project latent shards; VAE only decodes
# at inference. SDPA attention + optional torch.compile.
#
# Measured VRAM (4090 24G, 6-step probe): B128=11.5G  B192=16.2G  B256=21.5G
# Default B192 keeps ~1/3 headroom -> guaranteed no OOM. Paper effective batch
# is 16; if reproducing the paper recipe use BS=16 ACC=1.
# Run from baseline/FontDiffuser:  bash ../adapter/fontdiffuser/train_latent.sh
set -e
STEPS=${STEPS:-150000}
BS=${BS:-192}
ACC=${ACC:-1}
MP=${MP:-no}            # no | fp16
COMPILE=${COMPILE:-yes} # torch.compile unet+encoders (speed; VRAM ~= same here)
OUT=${OUT:-outputs/latent_top10}
CKPT=${CKPT:-10000}

accelerate launch --num_processes=1 --mixed_precision=$MP train_latent.py \
    --seed=123 \
    --experience_name="fontdiffuser_latent_top10" \
    --output_dir="$OUT" \
    --report_to="tensorboard" \
    --unet_channels=64,128,256,512 \
    --content_encoder_downsample_size=3 \
    --channel_attn=True \
    --content_start_channel=64 \
    --style_start_channel=64 \
    --train_batch_size=$BS \
    --perceptual_coefficient=0.01 \
    --offset_coefficient=0.5 \
    --max_train_steps=$STEPS \
    --ckpt_interval=$CKPT \
    --gradient_accumulation_steps=$ACC \
    --log_interval=50 \
    --learning_rate=1e-4 \
    --lr_scheduler="linear" \
    --lr_warmup_steps=10000 \
    --drop_prob=0.1 \
    --mixed_precision=$MP \
    $([ "$COMPILE" = "yes" ] && echo --compile)
