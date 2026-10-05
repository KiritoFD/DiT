#!/bin/bash
set -e

cd /root/Workspace/xy/DiT
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

RESULTS_DIR="exp/v37_skelnet_sp"
mkdir -p "${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/train.log"
echo "=== Starting SkelNet v37 DiT-Sp Heavy-Aug Real26k Training @ $(date) ===" | tee -a "${LOG_FILE}"

/opt/conda/envs/cu121/bin/python3 -u tools/train_skelnet_v37_sp_heavyaug.py \
    --csv assets/train_top10_style23_real.csv \
    --eval-cache data/top10_style23/eval_real200_cache.pt \
    --tgt-shards data/top10_style23/shards_gtskel_w7 \
    --cond-shards data/top10_style23/shards_std_w7 \
    --results-dir exp/v37_skelnet_sp \
    --experiment-name v37-sp-real26k \
    --batch 128 \
    --lr 3e-4 \
    --warmup 1500 \
    --max-steps 50000 \
    --log-every 50 \
    --ckpt-every 2500 \
    --eval-every 2500 \
    --lam-whiten 1.0 \
    --lam-dir 0.2 \
    --lam-mag 0.1 \
    --lam-rec 0.3 \
    --lam-lap 0.1 \
    --loss-ramp 1000 \
    --deform-prob 0.85 \
    --mask-prob 0.35 \
    --noise-prob 0.40 \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "Process launched in background with PID: ${PID}"
echo "Monitoring logs in ${LOG_FILE}..."
sleep 5
tail -n 25 "${LOG_FILE}"
