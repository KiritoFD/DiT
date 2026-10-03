#!/bin/bash
# v35 联合训练 (重做): 泄露已修 (pred 评测口径强制 β=1), 小 lr, 冻结 stage2。
# 训练器已带: eager 取样 + 无条件解码墨量校验 + 评测权重快照 (evalgen_step*.pt)。
# 铁律: 不允许 expandable_segments; 训练占卡期间不并发跑任何评测。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs assets/results/v35_joint
$PY -u tools/train_joint_stage1_stage2.py \
  --gen-ckpt assets/results/v33_stage1_xs/20261001-113156-v33-stage1-xs/checkpoints/0030000.pt \
  --bak-ckpt assets/results/v32_stage2_img/20261001-062933-v32-stage2-img/checkpoints/0080000.pt \
  --train-bak 0 --lam-skel 0 --batch 64 --gen-steps 8 \
  --lr 2e-6 \
  --compile-mode reduce-overhead \
  --max-steps 5000 --eval-every 500 --ckpt-every 500 \
  --out-dir assets/results/v35_joint 2>&1 | tee logs/v35_joint.log
