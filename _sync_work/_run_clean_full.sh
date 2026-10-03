#!/bin/bash
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT:$PYTHONPATH
echo "=== clean TRAIN (scheme B) ==="
/opt/conda/bin/python tools/clean_v2.py \
    --csv assets/train_fame.csv \
    --img-root data/imgs/final_imgs_256 \
    --out-root data/imgs/final_imgs_256_clean_v2 \
    --scheme B \
    --report assets/clean_report_train_B.csv \
    --workers 32 2>&1
echo ""
echo "=== clean EVAL (scheme B) ==="
/opt/conda/bin/python tools/clean_v2.py \
    --csv assets/eval_fame_strict.csv \
    --img-root data/imgs/final_imgs_256 \
    --out-root data/imgs/final_imgs_256_clean_v2 \
    --scheme B \
    --report assets/clean_report_eval_B.csv \
    --workers 32 2>&1
echo ""
echo "=== DONE ==="
ls assets/clean_report_train_B.csv assets/clean_report_eval_B.csv
echo "cleaned images: $(ls data/imgs/final_imgs_256_clean_v2/ 2>/dev/null | wc -l)"
