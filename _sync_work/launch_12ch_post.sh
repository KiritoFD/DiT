#!/bin/bash
# v13 12ch 后训练：从 base 的 155k ckpt 做 4ch->12ch 通道扩展，加 aux 通道继续训
#
# ⚠ 通道扩展必须走 --expand-from-4ch，**不能**用 --resume-full：
#   后者 strict=False，形状不匹配的 3 个张量会被静默跳过、停在随机初始化。
#
# ⚠ 不设 PYTORCH_CUDA_ALLOC_CONF —— 用默认分配器策略（expandable_segments 会让
#   reserved 每步波动；实测不带它也能跑 batch360）。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/v13_series/v13_12ch_post
mkdir -p "$LOGD"

CKPT=$(ls -t /root/Workspace/xy/DiT/assets/results/v13_base_50k/*/checkpoints/*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "[12ch-post] 找不到 base ckpt，退出"
    exit 1
fi
echo "[12ch-post] ===== START $(date) ====="
echo "[12ch-post] base ckpt = $CKPT"

TS=$(date +%Y%m%d-%H%M%S)
$PY -u src/train/train.py --config src/train/configs/v13_12ch_post.json \
    --expand-from-4ch "$CKPT" \
    2>&1 | tee "$LOGD/train_$TS.log"
echo "[12ch-post] ===== END rc=$? $(date) ====="
