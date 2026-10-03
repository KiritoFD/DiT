#!/bin/bash
# pivot_std_w7.sh — 停掉数据错配的旧训练, 用整卡全速造 shards_std_w7
#
# 错配: 训练是 (std 条件 3px -> w7 目标 7px), 模型被迫"一边形变一边加粗 2.3x",
#       实测输出宽度≈输入宽度(std 3.4px / 模型 2.6~3.7px / 目标 6.9px)。
# 修法: 条件也做成 7px -> 任务退化成纯形变(等宽)。
cd /root/Workspace/xy/DiT
echo "=== [1] 停掉旧训练与排队任务 ==="
pkill -f run_auxloss_arms
pkill -f train_skelnet_dit
pkill -f after_arms_beta
pkill -f build_std_w7
sleep 8
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo -n "残留 skelnet/build 进程: "; pgrep -c -f 'train_skelnet|build_std_w7' || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
FILT='Warning|warn|pytree|pkg_resources|preload|skel-guard'

echo "=== [2] 探针: 宽度对照 (std / 膨胀后 / w7 目标) ==="
$PY -u tools/build_std_w7.py --probe --limit 1 2>&1 | grep -vE "$FILT" | tail -12

echo "=== [3] 全量生成 shards_std_w7 (7px, batch 32, 整卡) ==="
$PY -u tools/build_std_w7.py --iters 3 --batch 32 2>&1 | grep -vE "$FILT" | tail -24
echo BUILD_STD_W7_DONE
