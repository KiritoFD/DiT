#!/bin/bash
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if [ -d "final_latents_aug" ] && [ -f "5script/train_3top30_aug.csv" ]; then
    echo "final_latents_aug already exists, skip augmentation"
    exit 0
fi
rm -rf final_latents_aug 5script/train_3top30_aug.csv 2>/dev/null
echo "Starting augmentation..."
/opt/conda/bin/python tools/augment_encode_latents.py \
    --train-csv 5script/train_3top30_nobeike.csv \
    --img-root final_imgs_256 \
    --latent-dir final_latents \
    --vae pretrained_models/sd-vae-ft-ema \
    --out-latent final_latents_aug \
    --out-csv 5script/train_3top30_aug.csv \
    --target 4 --shard-size 5000 \
    > /tmp/augment.log 2>&1
echo "Augmentation exit code: $?"
