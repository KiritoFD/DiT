#!/bin/bash
# 用**同一套 250 张 strict** 评多个 base ckpt，得到干净的 strict 曲线。
# 目的：判断过拟合是否已经伤到 strict（若 130k 就见顶 -> 155k 之后是浪费）
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
# 挑**含 0155000.pt 的那个**目录（残留的空 run 目录会干扰 ls | head -1）
CKPT_DIR=$(dirname $(ls -t /root/Workspace/xy/DiT/assets/results/v13_base_50k/*/checkpoints/0155000.pt 2>/dev/null | head -1))
echo "[sweep] ckpt dir = $CKPT_DIR"

for S in 80000 100000 120000 130000 140000 155000; do
    F="$CKPT_DIR/$(printf '%07d' $S).pt"
    [ -f "$F" ] || { echo "[sweep] 缺 $F，跳过"; continue; }
    echo "[sweep] ===== step $S $(date +%H:%M:%S) ====="
    PYTHONPATH=/root/Workspace/xy/DiT LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 \
      MASTER_ADDR=localhost MASTER_PORT=$((29700 + S % 1000)) \
      $PY -u src/train/train.py --config src/train/configs/v13_base_50k.json \
      --resume-full "$F" --eval-only --results-dir /tmp/_sweep_$S 2>&1 \
      | grep -E "eval-only|set=seen|set=strict|Error|Traceback" | tail -6
done
echo "[sweep] ===== DONE $(date +%H:%M:%S) ====="
