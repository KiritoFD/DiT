#!/bin/bash
# v15b + SupCon 重训的多风格表（唯一变量 = 词表）
#
# 对照:
#   原 v15b/v15a 用 assets/multistyle_k4_pretrained.pt  (DINO+K-Means, 余弦 0.860)
#   本跑   用 assets/multistyle_k4_supcon.pt            (SupCon,      余弦 0.0019)
#
# 参照基线:
#   v13 base 155k  strict = 0.5703
#   v15a 150k      strict = 0.5699   (旧表, 多风格零增益)
#   v15b 115k      strict = 0.5608   (旧表, 还在涨)
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/v15_series/v15b_supcon
mkdir -p "$LOGD"
TS=$(date +%Y%m%d-%H%M%S)

echo "[v15b-supcon] ===== START $(date) ====="
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29900 \
$PY -u src/train/train.py --config src/train/configs/v15b_supcon.json \
    --results-dir assets/results/v15b_supcon \
    2>&1 | tee "$LOGD/train_$TS.log"

echo "[v15b-supcon] ===== END rc=$? $(date) ====="
