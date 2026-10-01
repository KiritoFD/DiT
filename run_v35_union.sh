#!/usr/bin/env bash
set -e

ROOT="/root/Workspace/xy/DiT"
cd "$ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

LOG_FILE="$ROOT/logs/v35_union.log"
mkdir -p "$ROOT/logs" "$ROOT/exp/v35_union"

echo "=== 启动 v35 正统 1-Step 单步流投影端到端联合微调 (直接 Batch 128, 目标 10k, 每 1k 评测) ===" | tee -a "$LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "Stage 1 (Generator): exp/v31_stage1_skel @ 80,000 步" | tee -a "$LOG_FILE"
echo "Stage 2 (Backbone):  exp/v34_stage2_mix25 @ 30,000 步" | tee -a "$LOG_FILE"
echo "配置: 单卡直接 Batch 128, 目标 10,000 步, 评测/存档间隔 1,000 步" | tee -a "$LOG_FILE"

/opt/conda/envs/cu121/bin/python3 -u tools/train_joint_v35_1step.py \
    --gen-ckpt exp/v31_stage1_skel/20261001-005831-v31-stage1-skel/checkpoints/0080000.pt \
    --bak-ckpt exp/v34_stage2_mix25/20261001-220019-v34_stage2_mix25/checkpoints/0030000.pt \
    --results-dir exp/v35_union \
    --experiment-name v35-union-1step \
    --batch 128 \
    --accum-steps 1 \
    --gen-steps-eval 25 \
    --gen-lr 2e-5 \
    --lam-skel-flow 1.0 \
    --lam-skel-mse 0.3 \
    --glyph-mask-prob 0.1 \
    --max-steps 10000 \
    --warmup 500 \
    --log-every 20 \
    --ckpt-every 1000 \
    --eval-every 1000 \
    2>&1 | tee -a "$LOG_FILE"
