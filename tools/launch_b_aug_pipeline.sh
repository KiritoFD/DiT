#!/bin/bash
set -e

WORKDIR=/home/ds/Workspace/DiT
cd $WORKDIR

export PYTHONPATH=$WORKDIR
PYTHON=/home/ds/miniconda3/envs/pytorch/bin/python

LOG_DIR=$WORKDIR/experiments/capacity_ladder/logs
mkdir -p $LOG_DIR
mkdir -p $WORKDIR/experiments/capacity_ladder/results/tier3_b_aug

ENCODE_LOG=$LOG_DIR/encode_aug.log
TRAIN_LOG=$LOG_DIR/tier3_b_aug_train.log

echo "================================================================================"
echo "【Stage 1/2】开始 VAE Latent Shards 断点续编 (77,823 样本)..."
echo "================================================================================"
$PYTHON -u tools/encode_top10_aug.py > $ENCODE_LOG 2>&1

echo "================================================================================"
echo "【Stage 2/2】启动 Tier 3 (B/2) 数据增强大模型训练 (ETA = 10 Hours, 48,000 Steps)..."
echo "================================================================================"
CONFIG=$WORKDIR/experiments/capacity_ladder/configs/tier3_b_aug.json
$PYTHON -u src/train/train.py --config $CONFIG > $TRAIN_LOG 2>&1
