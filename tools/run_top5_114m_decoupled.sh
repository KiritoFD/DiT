#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

mkdir -p scratch
LOG_FILE="scratch/subsets_top5_114m_decoupled.log"
PYTHON="/opt/conda/envs/cu121/bin/python"

echo "=========================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Top 5 114M Decoupled (Exp A Top10 Architecture)"
echo "Config: src/unet/configs/subsets/top5_114m_decoupled_50k.json"
echo "Log: $LOG_FILE"
echo "=========================================================="

$PYTHON -u -m src.unet.train \
    --config src/unet/configs/subsets/top5_114m_decoupled_50k.json \
    2>&1 | tee "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Top 5 114M Decoupled Training Finished Successfully!"
