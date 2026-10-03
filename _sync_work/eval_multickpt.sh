#!/bin/bash
# 用同一套 250 张 strict 评多个 ckpt，画一条干净的 strict 曲线
# 目的：区分"过拟合已伤到 strict" vs "strict 还在涨"
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
# 只取 run 目录（名字含 v13-base-50k），且按时间取最新：
#   ls -d */ 会把 posters/ eval_samples_ctrl/ 也算进来；head -1 取的是字母序不是时间序
BASE=$(ls -dt assets/results/v13_base_50k/*v13-base-50k*/ | head -1)
echo "[multickpt] base run = $BASE"
echo "[multickpt] ===== START $(date) ====="

for S in 0100000 0130000 0155000; do
    CK="$BASE/checkpoints/$S.pt"
    if [ ! -f "$CK" ]; then
        echo "[multickpt] 缺 $CK，跳过"
        continue
    fi
    echo
    echo "[multickpt] ===== step $S $(date) ====="
    LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost \
        MASTER_PORT=$((29620 + 10#$S % 100)) \
        $PY -u src/train/train.py \
        --config src/train/configs/v13_base_50k.json \
        --resume-full "$CK" --eval-only \
        --results-dir "/tmp/_multi_$S" 2>&1 \
        | grep -E "eval-only|in-mem-eval\] step=|set=strict|set=seen|Traceback|Error"
done
echo
echo "[multickpt] ===== DONE $(date) ====="
