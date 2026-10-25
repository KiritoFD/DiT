#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

mkdir -p scratch/subsets_top2_top5
ORCH_LOG="scratch/subsets_top2_top5/orchestration.log"
PYTHON="/opt/conda/envs/cu121/bin/python"

echo "==========================================================" | tee -a "$ORCH_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Subsets Training Suite (Top 2 -> Top 5)" | tee -a "$ORCH_LOG"
echo "==========================================================" | tee -a "$ORCH_LOG"

# 1. Top 2 Training
echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> Stage 1: Launching Top 2 U100 Base (50,000 steps)..." | tee -a "$ORCH_LOG"
TOP2_LOG="scratch/subsets_top2_top5/top2_u100_base.log"
$PYTHON -u -m src.unet.train \
    --config src/unet/configs/subsets/u100_base_top2_noaug.json \
    > "$TOP2_LOG" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> Stage 1: Top 2 U100 Base Finished Successfully!" | tee -a "$ORCH_LOG"

# Cooldown & GPU memory check
sleep 10

# 2. Top 5 Training
echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> Stage 2: Launching Top 5 U100 Base (50,000 steps)..." | tee -a "$ORCH_LOG"
TOP5_LOG="scratch/subsets_top2_top5/top5_u100_base.log"
$PYTHON -u -m src.unet.train \
    --config src/unet/configs/subsets/u100_base_top5_noaug.json \
    > "$TOP5_LOG" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> Stage 2: Top 5 U100 Base Finished Successfully!" | tee -a "$ORCH_LOG"
echo "==========================================================" | tee -a "$ORCH_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All Subsets Experiments Completed!" | tee -a "$ORCH_LOG"
echo "==========================================================" | tee -a "$ORCH_LOG"
