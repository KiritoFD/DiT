#!/bin/bash
set -e

cd /root/Workspace/xy/DiT
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 杀死旧的进程以释放显存
pkill -9 -f train_render || true
pkill -9 -f train_joint || true
sleep 2

RESULTS_DIR="exp/v41_render_mix90"
mkdir -p "${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/train.log"
echo "=== Starting Render Model (90% Pred + 10% GT Mix, Batch 208, 20GB VRAM) @ $(date) ===" | tee "${LOG_FILE}"

/opt/conda/envs/cu121/bin/python3 -u tools/train_render_v41_mix90_20g.py \
    --csv assets/train_top10_style23_real.csv \
    --eval-cache data/top10_style23/eval_real200_cache.pt \
    --img-shards data/top10_style23/shards_img \
    --skel-shards-pred data/top10_style23/shards_predskel_v31 \
    --skel-shards-gt data/top10_style23/shards_aux_skel3 \
    --skelnet-ckpt exp/v38_skelnet_s2_pure/20261002-142735-v38-s2-pure-b1440-20g/checkpoints/best_ssim.pt \
    --init-render-ckpt exp/v39_render_mix50/20261002-200307-v39-render-mix50-b208-20g/checkpoints/best_ssim.pt \
    --results-dir exp/v41_render_mix90 \
    --experiment-name v41-render-mix90-b208-20g \
    --batch 208 \
    --lr 2e-4 \
    --warmup 500 \
    --max-steps 10000 \
    --deform-prob 0.5 \
    --deform-scale 1.0 \
    --log-every 25 \
    --ckpt-every 1000 \
    --eval-every 1000 \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "Render 90% Pred training process launched in background with PID: ${PID}"
echo "Monitoring logs in ${LOG_FILE}..."
sleep 10
tail -n 25 "${LOG_FILE}"
