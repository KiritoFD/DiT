#!/bin/bash
# run_self_cond_eval.sh - 在 4090 上对 v11_pretrain_M432_adaln4_sym 进行自条件推理 A/B 评估
set -e

cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor

PYTHON="/opt/conda/envs/cu121/bin/python"

CKPT="assets/results/v11_pretrain_M432_adaln4_sym/20260913-164820-v11-pretrain-M432-adaln4-sym/checkpoints/0042500.pt"
CONFIG="src/train/configs/v11_pretrain_M432_adaln4_sym.json"
SKEL_DIR="data/skel/std_skel3_latents_fame_sym"

echo "============================================================"
echo " [1/2] Seen 集 (n=10) 自条件推理 A/B 评测"
echo "============================================================"
$PYTHON tools/eval/eval_self_cond.py \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --eval-csv "assets/eval_seen_v10.csv" \
    --skel-dir "$SKEL_DIR" \
    --n 10 \
    --device cuda \
    --cfg-scale 0.7 \
    --eval-steps 50 \
    --first-pass-steps 20 \
    --blend-alphas "0.0,0.3,0.5" \
    --batch 10

echo ""
echo "============================================================"
echo " [2/2] Strict 集 (n=50) 自条件推理 A/B 评测"
echo "============================================================"
$PYTHON tools/eval/eval_self_cond.py \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --eval-csv "assets/eval_fame3_strict50.csv" \
    --skel-dir "$SKEL_DIR" \
    --n 50 \
    --device cuda \
    --cfg-scale 0.7 \
    --eval-steps 50 \
    --first-pass-steps 20 \
    --blend-alphas "0.0,0.3,0.5" \
    --batch 10

echo ""
echo ">>> All evaluations finished successfully!"
