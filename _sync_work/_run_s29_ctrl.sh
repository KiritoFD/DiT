#!/bin/bash
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT:$PYTHONPATH
mkdir -p assets/results/s29_ctrl_gt_skel_1px
exec /opt/conda/bin/python src/train/train_controlnet.py --config src/train/configs/s29_ctrl_gt_skel_1px.json
