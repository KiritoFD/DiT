#!/bin/bash
# v13 串行链：base(250k) -> 12ch 后训练
#
# 阶段 1: v13 base 4ch 预训练（新 50k 数据集，adaLN4），250k 步
# 阶段 2: 从阶段 1 的最后一个 ckpt 做 **4ch->12ch 通道扩展**，加 aux 通道继续训
#
# ⚠ 通道扩展必须走 --expand-from-4ch，**不能**用 --resume-full：
#   后者 strict=False，形状不匹配的 3 个张量会被静默跳过、停在随机初始化。
#
# ⚠ 不设 PYTORCH_CUDA_ALLOC_CONF —— 用默认分配器策略。
#   (expandable_segments 会让 reserved 每步波动；实测不带它也能跑 batch360)
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/v13_series
mkdir -p "$LOGD"

echo "[v13-serial] ===== START $(date) ====="

# ── 阶段 1: base 4ch ────────────────────────────────────────────────
TS=$(date +%Y%m%d-%H%M%S)
echo "[v13-serial] ===== STAGE 1: v13 base 4ch (250k) $(date) ====="
$PY -u src/train/train.py --config src/train/configs/v13_base_50k.json \
    2>&1 | tee "$LOGD/v13_base_50k/train_$TS.log"
RC1=$?
echo "[v13-serial] ===== STAGE 1 END rc=$RC1 $(date) ====="
if [ $RC1 -ne 0 ]; then
    echo "[v13-serial] 阶段 1 失败，不进入阶段 2"
    exit $RC1
fi

# ── 找到阶段 1 的最后一个 ckpt ───────────────────────────────────────
CKPT=$(ls -t /root/Workspace/xy/DiT/assets/results/v13_base_50k/*/checkpoints/*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "[v13-serial] 找不到 base ckpt，退出"
    exit 1
fi
echo "[v13-serial] 用 base ckpt: $CKPT"

# ── 阶段 2: 12ch 后训练 ─────────────────────────────────────────────
TS=$(date +%Y%m%d-%H%M%S)
echo "[v13-serial] ===== STAGE 2: 12ch 后训练 $(date) ====="
$PY -u src/train/train.py --config src/train/configs/v13_12ch_post.json \
    --expand-from-4ch "$CKPT" \
    2>&1 | tee "$LOGD/v13_12ch_post/train_$TS.log"
echo "[v13-serial] ===== STAGE 2 END rc=$? $(date) ====="
echo "[v13-serial] ===== ALL DONE $(date) ====="
