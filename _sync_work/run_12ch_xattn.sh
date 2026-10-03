#!/bin/bash
# v12 系列第二段: 12ch -> xattn  (串行, tmux 内运行)
#   两者都基于**已修复**的代码:
#     - dit.py: factorized_cat 加入 drop guard 元组 (原先零条件 dropout -> CFG 分支未训练)
#     - dit.py: xattn 新增 q_pos 开关 (原先 Q 无位置 -> "空间寻址"退化成"内容寻址")
#     - train.py: aux 权重缺失时 raise / in_channels 一致性断言
#   batch 全部对齐 v12 基线的 360, 保证"同 step == 同数据量"可比。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
CFG=src/train/configs

for NAME in v12_12ch v12_xattn; do
    LOGD=/root/Workspace/xy/DiT/logs/v12_series/$NAME
    mkdir -p "$LOGD"
    TS=$(date +%Y%m%d-%H%M%S)
    echo "[series2] ===== START $NAME  $(date) ====="
    $PY -u src/train/train.py --config $CFG/${NAME}_pretrain.json 2>&1 | tee "$LOGD/train_$TS.log"
    echo "[series2] ===== END $NAME  rc=$?  $(date) ====="
done
echo "[series2] ALL DONE $(date)"
