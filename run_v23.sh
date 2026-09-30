#!/usr/bin/env bash
# v23-splitnorm: 单变量 A/B —— 只改 cond_fusion 归一化方式 (joint -> split)
# 其余逐项等于 v22_std_callig_aug (from scratch, 150k steps, deform_skel_v10 + stroke_mod)
set -eo pipefail

export PYTHONPATH="/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/Workspace/xy/DiT

mkdir -p logs assets/results/v23_splitnorm

PY=/opt/conda/envs/cu121/bin/python

echo "[run_v23] 启动 v23-splitnorm (cond_fusion_norm=split)..."
echo "[run_v23] 修复: 各操作数独立 LayerNorm 再 cat, 抹平 |e_callig|=9.86 vs |e_glyph_vec|=60.90 的 38x 方差失衡"
echo "[run_v23] 实测: callig 半边振幅 0.186 -> 0.800 (4.30x), callig/glyph 比 0.167 -> 1.001"

$PY -m src.train.train --config src/train/configs/v23_splitnorm.json \
    > logs/v23_splitnorm.log 2>&1
