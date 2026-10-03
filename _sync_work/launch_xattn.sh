#!/bin/bash
# 启动 v12_xattn（已修: xattn_q_pos=true + batch 360 对齐基线）
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
CFG=src/train/configs/v12_xattn_pretrain.json
LOGD=/root/Workspace/xy/DiT/logs/v12_series/v12_xattn
mkdir -p "$LOGD"
TS=$(date +%Y%m%d-%H%M%S)
echo "[xattn] ===== START $(date) ====="
$PY -u src/train/train.py --config "$CFG" 2>&1 | tee "$LOGD/train_$TS.log"
echo "[xattn] ===== END rc=$? $(date) ====="
