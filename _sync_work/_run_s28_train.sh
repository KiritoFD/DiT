#!/bin/bash
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT:$PYTHONPATH
mkdir -p assets/results/s28_std_dino_pretrain
exec /opt/conda/bin/python src/train/train.py --config src/train/configs/s28_std_dino_pretrain.json
