#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

mkdir -p scratch
LOG_FILE="scratch/subsets_top5_u100_deep_sym4.log"
PYTHON="/opt/conda/envs/cu121/bin/python"

echo "=========================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Pure Deepened U100 (Top 5, 114M, num_res_blocks=4, NO Decoupling)"
echo "Config: src/unet/configs/subsets/top5_u100_deep_sym4_50k.json"
echo "Log: $LOG_FILE"
echo "=========================================================="

$PYTHON -u -m src.unet.train \
    --config src/unet/configs/subsets/top5_u100_deep_sym4_50k.json \
    2>&1 | tee "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Top 5 U100 Deep Sym4 Training Finished Successfully!"
