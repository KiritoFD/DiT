#!/bin/bash
# 过拟合验证 (v36 union): 训练集 = strict84 (与 eval 同源), 看模型能不能背下来。
# 判据: 1) L 明显下降; 2) eval 的 strict84_e2e ssim 向 oracle 靠拢;
#       3) 重载 ckpt 推理, 生成骨架解码有墨且逼近目标。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs assets/results/overfit_v36
$PY -u tools/train_joint_v36.py \
  --overfit --batch 16 --gen-lr 1e-4 \
  --max-steps 600 --eval-every 200 --ckpt-every 200 --log-every 20 \
  --results-dir assets/results/overfit_v36 \
  2>&1 | tee logs/overfit_v36.log
