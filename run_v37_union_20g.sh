#!/bin/bash
set -e

cd /root/Workspace/xy/DiT
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

RESULTS_DIR="exp/v37_union_20g"
mkdir -p "${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/train.log"
echo "=== Starting 1-Step Joint Training v37 (Batch 240, 20.05 GB VRAM) @ $(date) ===" | tee -a "${LOG_FILE}"

/opt/conda/envs/cu121/bin/python3 -u tools/train_joint_v37_1step_20g.py \
    --gen-ckpt exp/v37_skelnet_sp/20261002-123800-v37-sp-real26k/checkpoints/0007500.pt \
    --bak-ckpt exp/v34_stage2_mix25/20261001-220019-v34_stage2_mix25/checkpoints/0030000.pt \
    --csv assets/train_top10_style23_real.csv \
    --eval-cache data/top10_style23/eval_real200_cache.pt \
    --shards-img data/top10_style23/shards_img \
    --shards-std data/top10_style23/shards_std_w7 \
    --shards-gt data/top10_style23/shards_gtskel_w7 \
    --results-dir exp/v37_union_20g \
    --experiment-name v37-union-b240-20g \
    --batch 240 \
    --gen-lr 2e-5 \
    --warmup 500 \
    --max-steps 15000 \
    --log-every 25 \
    --ckpt-every 1000 \
    --eval-every 1000 \
    --lam-skel-flow 1.0 \
    --lam-skel-mse 0.3 \
    --lam-skel-lap 0.1 \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "Process launched in background with PID: ${PID}"
echo "Monitoring logs in ${LOG_FILE}..."
sleep 5
tail -n 25 "${LOG_FILE}"
