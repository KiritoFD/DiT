#!/usr/bin/env bash
set -e

WORKSPACE="/home/ds/Workspace/moyi"
cd "${WORKSPACE}"

export PYTHONPATH="${WORKSPACE}/ref/moyi:${WORKSPACE}/ref/moyi/moyun:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/ds/miniconda3/envs/pytorch/bin/python"

RESULTS_DIR="${WORKSPACE}/results/moyi_top10_rf"
mkdir -p "${RESULTS_DIR}/logs"
LOG_FILE="${RESULTS_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "Starting Moyi / Moyun Top10 Rectified Flow Training"
echo "Results Dir: ${RESULTS_DIR}"
echo "Log file:    ${LOG_FILE}"
echo "============================================================"

nohup ${PYTHON} -u "${WORKSPACE}/ref/moyi/train_moyi_top10_repro.py" \
    --model moyun-12channel-B \
    --batch-size 112 \
    --lr 1e-4 \
    --max-steps 80000 \
    --log-every 10 \
    --eval-every 1000 \
    --ckpt-every 5000 \
    --num-workers 4 \
    --results-dir "${RESULTS_DIR}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Process launched in background! PID = ${PID}"
echo "${PID}" > "${RESULTS_DIR}/train.pid"
echo "Run 'tail -f ${LOG_FILE}' to view live output."
