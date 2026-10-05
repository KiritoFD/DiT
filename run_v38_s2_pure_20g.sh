#!/bin/bash
set -e

cd /root/Workspace/xy/DiT
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

RESULTS_DIR="exp/v38_skelnet_s2_pure"
mkdir -p "${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/train.log"
echo "=== Starting SkelNet v38 S/2 Pure w7 Training (Batch 1440, 20GB VRAM) @ $(date) ===" | tee -a "${LOG_FILE}"

/opt/conda/envs/cu121/bin/python3 -u tools/train_skelnet_v38_pure_s2_20g.py \
    --csv assets/train_top10_style23_real.csv \
    --eval-cache data/top10_style23/eval_real200_cache.pt \
    --tgt-shards data/top10_style23/shards_gtskel_w7 \
    --cond-shards data/top10_style23/shards_std_w7 \
    --results-dir exp/v38_skelnet_s2_pure \
    --experiment-name v38-s2-pure-b1440-20g \
    --batch 1440 \
    --lr 5e-4 \
    --warmup 1000 \
    --max-steps 20000 \
    --log-every 25 \
    --ckpt-every 1000 \
    --eval-every 1000 \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "Process launched in background with PID: ${PID}"
echo "Monitoring logs in ${LOG_FILE}..."
sleep 5
tail -n 25 "${LOG_FILE}"
