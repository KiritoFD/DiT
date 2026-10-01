#!/usr/bin/env bash
set -e

ROOT="/root/Workspace/xy/DiT"
cd "$ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

LOG_STAGE1="$ROOT/logs/v36_stage1_skel_w7.log"
LOG_UNION="$ROOT/logs/v36_union_w7.log"
mkdir -p "$ROOT/logs" "$ROOT/exp/v36_stage1_skel_w7" "$ROOT/exp/v36_union_w7"

echo "============================================================" | tee -a "$LOG_STAGE1"
echo "=== [阶段一] 启动 SkelNet v2 (w7 粗骨架强基模) 预训练 ===" | tee -a "$LOG_STAGE1"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_STAGE1"
echo "配置: src/train/configs/v36_stage1_skel_w7.json (Batch 384, max 60k, early_stop 5)" | tee -a "$LOG_STAGE1"
echo "============================================================" | tee -a "$LOG_STAGE1"

/opt/conda/envs/cu121/bin/python3 -u src/train/train.py \
    --config src/train/configs/v36_stage1_skel_w7.json \
    2>&1 | tee -a "$LOG_STAGE1"

echo "============================================================" | tee -a "$LOG_STAGE1"
echo "=== [阶段一完成] SkelNet v2 预训练结束，寻找最佳/最终 Checkpoint ===" | tee -a "$LOG_STAGE1"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_STAGE1"
echo "============================================================" | tee -a "$LOG_STAGE1"

# 查找最新的 run 目录
RUN_DIR=$(ls -td exp/v36_stage1_skel_w7/2026* 2>/dev/null | head -n 1)
if [ -z "$RUN_DIR" ]; then
    echo "错误: 未找到 exp/v36_stage1_skel_w7 下的运行目录!" | tee -a "$LOG_STAGE1"
    exit 1
fi

# 寻找最新或最佳 Checkpoint
BEST_CKPT="$RUN_DIR/checkpoints/best_checkpoint.pt"
if [ ! -f "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -t $RUN_DIR/checkpoints/*.pt 2>/dev/null | grep -v 'opt' | head -n 1)
fi
echo "选定 Stage 1 检查点: $BEST_CKPT" | tee -a "$LOG_STAGE1"

echo "============================================================" | tee -a "$LOG_UNION"
echo "=== [阶段二] 自动拉起 1-Step Batch-128 端到端联合微调 ===" | tee -a "$LOG_UNION"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_UNION"
echo "Stage 1 (Generator): $BEST_CKPT" | tee -a "$LOG_UNION"
echo "Stage 2 (Backbone):  exp/v34_stage2_mix25/20261001-220019-v34_stage2_mix25/checkpoints/0030000.pt" | tee -a "$LOG_UNION"
echo "配置: 单卡直接 Batch 128, 目标 10,000 步, 骨架条件升级为 w7" | tee -a "$LOG_UNION"
echo "============================================================" | tee -a "$LOG_UNION"

/opt/conda/envs/cu121/bin/python3 -u tools/train_joint_v35_1step.py \
    --gen-ckpt "$BEST_CKPT" \
    --bak-ckpt exp/v34_stage2_mix25/20261001-220019-v34_stage2_mix25/checkpoints/0030000.pt \
    --results-dir exp/v36_union_w7 \
    --experiment-name v36-union-w7 \
    --shards-std data/top10_style23/shards_std_w7 \
    --shards-gt data/top10_style23/shards_gtskel_w7 \
    --shards-std-eval data/top10_style23/predskel_std84_e2e \
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
    2>&1 | tee -a "$LOG_UNION"

echo "🎉 [全部流水线圆满成功] SkelNet v2 预训练 + 端到端联合微调全流程闭环!" | tee -a "$LOG_UNION"
