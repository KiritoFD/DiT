#!/bin/bash
# runner_tj_cpu.sh - fame-tj-kxl tongji assets encode, CPU single-thread, sequential.
set -x
cd /root/Workspace/xy/DiT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/opt/conda/envs/cu121/bin/python

# 1) std skel latents (g condition; simkai for kai/xing, simli for li; cascade fallback)
$PY -u tools/data/build_std_skel1_latents.py \
    --csv assets/train_fame_tj_kxl.csv \
    --out-shards data/skel/std_skel3_latents_tj \
    --out-bank data/skel/skel_bank_std1_tj.npz \
    --dilate 1 --font-dir tools/fonts \
    --batch 8 --device cpu --threads 1 \
    --vae-path data/pretrained/pretrained_models/sd-vae-ft-ema \
    > logs/std_skel_tj_cpu.log 2>&1

# 2) aux canny/skel3 latents (from GT images, 8976 rows)
$PY -u tools/data/build_aux_latents_fame_e.py \
    --csv assets/train_tongji_only.csv \
    --img-root '' \
    --out-skel data/skel/aux_skel3_latents_tj \
    --out-canny data/aux/aux_canny_latents_tj \
    --batch 8 --skel-dilate 1 --workers 1 \
    --device cpu --vae-path data/pretrained/pretrained_models/sd-vae-ft-ema \
    > logs/aux_tj_cpu.log 2>&1

# 3) image latents (8976 rows)
$PY -u tools/data/build_image_latents.py \
    --csv assets/train_tongji_only.csv \
    --out data/latents/final_latents_tj_shards \
    --batch 8 --device cpu \
    --vae-path data/pretrained/pretrained_models/sd-vae-ft-ema \
    > logs/image_latents_tj_cpu.log 2>&1

echo "ALL DONE" > logs/tj_cpu_done.txt
