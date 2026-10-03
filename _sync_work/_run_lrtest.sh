#!/bin/bash
# LR range test: 4 个恒定 LR 各 2000 步, 从 175000 ckpt, callig_spatial 模型, 增强数据
set -u
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
CKPT=/root/Workspace/xy/DiT/assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-173200-v10b-stdskel-fame3-c41x-cos-e/checkpoints/0175000.pt
for LR in 5e-6 1.5e-5 5e-5 1.5e-4; do
  echo "=== LR=$LR ==="
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  $PY -u src/train/train.py --config src/train/configs/_lrtest_$LR.json \
      --resume-full $CKPT --fresh-scheduler 1 --resume-lr $LR \
      2>&1 | tee logs/lrtest_$LR.log | grep -E 'step=|Traceback|Error|Out of memory' | tail -30
done
echo "=== ALL DONE ==="
