#!/bin/bash
set -e

cd /root/Workspace/xy/DiT
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 杀死旧的进程以释放显存
pkill -9 -f train_render || true
sleep 2

RESULTS_DIR="exp/v39_render_mix50"
mkdir -p "${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/train.log"
echo "=== Starting Render Model (Stage 2) 50% Mix Training (Batch 208, No Ckpt, 20GB VRAM) @ $(date) ===" | tee "${LOG_FILE}"

/opt/conda/envs/cu121/bin/python3 -u tools/train_render_v39_mix50_20g.py \
    --csv assets/train_top10_style23_real.csv \
    --eval-cache data/top10_style23/eval_real200_cache.pt \
    --img-shards data/top10_style23/shards_img \
    --skel-shards-gt data/top10_style23/shards_aux_skel3 \
    --skel-shards-pred data/top10_style23/shards_predskel_v31 \
    --results-dir exp/v39_render_mix50 \
    --experiment-name v39-render-mix50-b208-20g \
    --batch 208 \
    --lr 5e-4 \
    --warmup 1000 \
    --max-steps 20000 \
    --deform-prob 0.5 \
    --deform-scale 1.0 \
    --log-every 25 \
    --ckpt-every 1000 \
    --eval-every 1000 \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "Render training process launched in background with PID: ${PID}"
echo "Monitoring logs in ${LOG_FILE}..."
sleep 10
tail -n 25 "${LOG_FILE}"
