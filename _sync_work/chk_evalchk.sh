#!/bin/bash
# 对账实验: 从"白图"ckpt 续跑, eager(不编译), eval 每 50 步, 看 eval 自己报的 gen_ink
# 以及它存下的 evalgen_step*.pt。用来判定"评测用的权重"与"保存的权重"是否一致。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs assets/results/v34_evalchk
$PY -u tools/train_joint_stage1_stage2.py \
  --gen-ckpt assets/results/v33_stage1_xs/20261001-113156-v33-stage1-xs/checkpoints/0030000.pt \
  --resume-gen assets/results/v34_e2e/joint_005000.pt \
  --bak-ckpt assets/results/v32_stage2_img/20261001-062933-v32-stage2-img/checkpoints/0080000.pt \
  --train-bak 0 --lam-skel 0 --batch 32 --gen-steps 8 --lr 1e-6 \
  --max-steps 100 --eval-every 50 --ckpt-every 0 \
  --out-dir assets/results/v34_evalchk 2>&1 | tee logs/v34_evalchk.log
