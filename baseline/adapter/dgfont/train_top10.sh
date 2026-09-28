#!/bin/bash
# DG-Font @ top10_style23, 256x256, 24 domains (23 slots + content_deng).
# Faithful losses otherwise (adv + GP + rec 0.1 + vec 0.01 + offset-norm 0.5).
# Measured VRAM (4090 24G, 25-iter probe, no expandable_segments):
#   B4=9.6G  B8=16.6G  B16=OOM  ->  default B8 keeps ~1/3 headroom (no OOM).
# Run from baseline/DG-Font:  bash ../adapter/dgfont/train_top10.sh
set -e
PY=${PY:-/opt/conda/envs/baseline/bin/python}
STEPS=${STEPS:-100000}     # epochs*iters
ITERS=${ITERS:-1000}
EPOCHS=$((STEPS / ITERS))
BS=${BS:-8}

python=$PY $PY main.py \
    --gpu 0 \
    --img_size 256 \
    --data_path ../data/dgfont/train \
    --output_k 24 \
    --val_num 5 \
    --sty_dim 128 \
    --batch_size $BS \
    --workers 8 \
    --epochs $EPOCHS \
    --iters $ITERS \
    --log_step 100 \
    --model_name GAN_top10_256 \
    --w_gp 10.0 --w_rec 0.1 --w_adv 1.0 --w_vec 0.01 --w_off 0.5
