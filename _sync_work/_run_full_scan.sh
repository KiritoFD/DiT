#!/bin/bash
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT:$PYTHONPATH
echo "=== scan TRAIN ==="
/opt/conda/bin/python tools/scan_image_pollution.py \
    --csv assets/train_fame.csv \
    --img-root data/imgs/final_imgs_256 \
    --out assets/scan_train_pollution.csv \
    --workers 32 2>&1
echo ""
echo "=== scan EVAL ==="
/opt/conda/bin/python tools/scan_image_pollution.py \
    --csv assets/eval_fame_strict.csv \
    --img-root data/imgs/final_imgs_256 \
    --out assets/scan_eval_pollution.csv \
    --workers 32 2>&1
echo ""
echo "=== DONE ==="
ls -la assets/scan_train_pollution.csv assets/scan_eval_pollution.csv
