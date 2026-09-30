#!/usr/bin/env bash
set -eo pipefail

export PYTHONPATH="/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/Workspace/xy/DiT

mkdir -p logs assets/results/v21_skelnet_200k

PY=/opt/conda/envs/cu121/bin/python

echo "[run_v21] 启动下游 200k 训练 (v21-skelnet-200k, deform_lr_scale=0.1, v10 ckpt)..."

$PY -m src.train.train --config src/train/configs/v21_skelnet_200k.json \
    > logs/v21_skelnet_200k.log 2>&1
