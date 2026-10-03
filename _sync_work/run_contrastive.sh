#!/bin/bash
# run_contrastive.sh — 等 DINO 提取完成后跑对比微调
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
while [ ! -f /tmp/dino_cls_train.npz ]; do sleep 10; done
echo "[$(date '+%F %T')] features ready, run contrastive"
/opt/conda/envs/cu121/bin/python /tmp/contrastive_finetune.py 2>&1 | tee /tmp/contrastive.log
echo "[$(date '+%F %T')] done"