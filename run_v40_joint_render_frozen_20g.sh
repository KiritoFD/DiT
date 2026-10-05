#!/bin/bash
set -e

cd /root/Workspace/xy/DiT
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 杀死旧的进程以释放显存
pkill -9 -f train_joint || true
pkill -9 -f train_render || true
sleep 2

RESULTS_DIR="exp/v40_joint_render_frozen"
mkdir -p "${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/train.log"
echo "=== Starting 1-Step Joint Training (Stage 2 Render Frozen, Batch 120, 20GB VRAM) @ $(date) ===" | tee "${LOG_FILE}"

/opt/conda/envs/cu121/bin/python3 -u tools/train_joint_v40_render_frozen_20g.py \
    --csv assets/train_top10_style23_real.csv \
    --eval-cache data/top10_style23/eval_real200_cache.pt \
    --skel-ckpt exp/v38_skelnet_s2_pure/20261002-142735-v38-s2-pure-b1440-20g/checkpoints/best_ssim.pt \
    --render-ckpt exp/v39_render_mix50/20261002-200307-v39-render-mix50-b208-20g/checkpoints/best_ssim.pt \
    --results-dir exp/v40_joint_render_frozen \
    --experiment-name v40-joint-render-frozen-b120-20g \
    --batch 120 \
    --lr 2e-5 \
    --warmup 500 \
    --max-steps 10000 \
    --lam-flow 1.0 \
    --lam-mse 0.3 \
    --lam-lap 0.1 \
    --lam-render 1.0 \
    --log-every 25 \
    --ckpt-every 1000 \
    --eval-every 1000 \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "Joint training process launched in background with PID: ${PID}"
echo "Monitoring logs in ${LOG_FILE}..."
sleep 10
tail -n 25 "${LOG_FILE}"
