#!/bin/bash
# v16 few-shot 串行长训：一行命令跑一个主题（lr 1e-5 / batch 384 / +50k 步 / 每 2500 步 eval）。
# 用法: bash _sync_work/run_v16.sh <主题>
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/v16_series
mkdir -p "$LOGD"

t=$1
CFG="src/train/configs/v16_fs_${t}.json"
[ -f "$CFG" ] || { echo "[v16] ✗ 缺 $CFG"; exit 1; }
CK=$(ls -t assets/results/v15a_multistyle_k4/*/checkpoints/0150000.pt | head -1)
[ -n "$CK" ] || { echo "[v16] ✗ 找不到 v15a 150k ckpt"; exit 1; }
OUT="assets/results/v16_fs_${t}_$(date +%m%d-%H%M%S)"
LOG="$LOGD/${t}_$(date +%m%d-%H%M%S).log"
echo "[v16] >>> $t  ckpt=$(basename $(dirname $(dirname $CK)))  ($(date '+%m-%d %H:%M'))"
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=${PORT:-29880} \
  $PY -u src/train/train.py --config "$CFG" --resume-full "$CK" \
      --fresh-scheduler true --train-only-new-callig --init-new-callig row_pt \
      --results-dir "$OUT" > "$LOG" 2>&1
rc=$?
grep -E "fresh-scheduler|set=fewshot|train-only-new-callig|风格表扩行|weight_decay" "$LOG" | tail -6
echo "[v16] <<< $t rc=$rc  ($(date '+%m-%d %H:%M'))  log=$LOG"
