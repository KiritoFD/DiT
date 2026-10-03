#!/bin/bash
# run_glyph_retrieval.sh — 训练间隙跑字形检索准确率 (raw vs centered), 决定 v9 是否用 centering
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
PY=/opt/conda/envs/cu121/bin/python
$PY tools/eval_glyph_retrieval.py \
    --build-csv assets/train_fame_clean_v8.csv \
    --query-csv assets/eval_fame_strict_clean_v8.csv \
    --img-root data/imgs/final_imgs_fame_v8 \
    --cache /tmp/dino_feat_v8.npz \
    --mode both --limit 300 2>&1 | tail -30