#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== clean csv files ==="
ls -la assets/train_fame_clean.csv assets/eval_fame_strict_clean.csv 2>/dev/null
echo "=== clean img dir ==="
ls -d data/imgs/final_imgs_256_clean 2>/dev/null && ls data/imgs/final_imgs_256_clean | head -3 && ls data/imgs/final_imgs_256_clean | wc -l
echo "=== clean latent dirs ==="
ls -d data/latents/final_latents_fame_clean 2>/dev/null || echo "(no clean latent dir)"
find . -maxdepth 1 -name '*latent*clean*' -o -maxdepth 1 -name '*clean*latent*' 2>/dev/null
echo "=== existing latent build script ==="
ls tools/build_*latent*.py 2>/dev/null
echo "=== train_fame_clean.csv head ==="
head -3 assets/train_fame_clean.csv 2>/dev/null
echo "=== line count ==="
wc -l assets/train_fame_clean.csv assets/eval_fame_strict_clean.csv 2>/dev/null
