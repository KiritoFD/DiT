#!/bin/bash
# 续训 (拉长 v34_e2e): 从 joint_005000.pt 的 **EMA 权重**接上, 再跑 10000 步。
# 配方与上一轮完全一致 (λ=0 纯端到端 / batch 64 / bf16 / compile reduce-overhead),
# 唯一变化 = 更长。铁律: 不允许 expandable_segments。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs assets/results/v34_e2e2
$PY -u tools/train_joint_stage1_stage2.py \
  --gen-ckpt assets/results/v33_stage1_xs/20261001-113156-v33-stage1-xs/checkpoints/0030000.pt \
  --resume-gen assets/results/v34_e2e/joint_005000.pt \
  --bak-ckpt assets/results/v32_stage2_img/20261001-062933-v32-stage2-img/checkpoints/0080000.pt \
  --train-bak 0 --lam-skel 0 --batch 64 --gen-steps 8 --lr 1e-5 \
  --compile-mode reduce-overhead \
  --max-steps 10000 --eval-every 2000 --ckpt-every 1000 \
  --out-dir assets/results/v34_e2e2 2>&1 | tee logs/v34_e2e2.log
