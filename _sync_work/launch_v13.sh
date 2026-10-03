#!/bin/bash
# v13 base（新 50k 数据集，4ch，adaLN4）
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
# 不设 PYTORCH_CUDA_ALLOC_CONF —— 用 PyTorch 默认分配器策略。
# (expandable_segments 会让 reserved 每步波动；v12 不带它也能跑 batch360)
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/v13_series/v13_base_50k
mkdir -p "$LOGD"
TS=$(date +%Y%m%d-%H%M%S)
echo "[v13] ===== START $(date) ====="
$PY -u src/train/train.py --config src/train/configs/v13_base_50k.json \
    2>&1 | tee "$LOGD/train_$TS.log"
echo "[v13] ===== END rc=$? $(date) ====="
