#!/bin/bash
# 端到端联合训练 (λ=0): 生成器唯一的监督 = 最终真迹图的 flow loss 反传。
# 不加载 shards_gt, 行集 = csv 全部行, 数据集一体返回 latent/skel_latent/y_callig。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
# 铁律: 不允许 expandable_segments (不要设 PYTORCH_CUDA_ALLOC_CONF)
mkdir -p logs assets/results/v34_e2e
$PY -u tools/train_joint_stage1_stage2.py \
  --gen-ckpt assets/results/v33_stage1_xs/20261001-113156-v33-stage1-xs/checkpoints/0030000.pt \
  --bak-ckpt assets/results/v32_stage2_img/20261001-062933-v32-stage2-img/checkpoints/0080000.pt \
  --train-bak 0 --lam-skel 0 --batch 64 --gen-steps 8 --lr 1e-5 \
  --compile-mode reduce-overhead \
  --max-steps 5000 --eval-every 1000 --ckpt-every 500 \
  --out-dir assets/results/v34_e2e 2>&1 | tee logs/v34_e2e.log
