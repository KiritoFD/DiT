#!/bin/bash
# 重建标准字形 1px 骨架 latent（级联字体，100% 覆盖）
cd /root/Workspace/xy/DiT || exit 1
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
/opt/conda/bin/python /root/Workspace/xy/DiT/tools/data/build_data/skel/std_skel1_latents.py \
    --csv assets/train_fame_clean_v8.csv \
    --font-dir tools/fonts \
    --out-shards data/skel/std_skel1_latents_fame_v8 \
    --out-bank data/skel/skel_bank_std1_v8.npz \
    > /tmp/build_data/skel/std_skel_v8.log 2>&1
echo "EXIT_CODE=$?" >> /tmp/build_data/skel/std_skel_v8.log
