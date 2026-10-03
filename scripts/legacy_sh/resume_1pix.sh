#!/bin/bash
cd /root/Workspace/xy/DiT
tmux new-session -d -s fame_1pix_ctrl "/opt/conda/bin/python -m src.train.train_controlnet --config src/train/configs/ctrl_fame_1pix_v1.json --attn-impl eager --resume assets/results/ctrl_fame_1pix_v1/20260830-185731-fame-ctrl-skel-1px-v1/checkpoints/0022500.pt > /tmp/fame_1pix_ctrl.log 2>&1"
tmux new-session -d -s 1pix_daemon "/opt/conda/bin/python -u src/eval/eval_ctrl_metrics_daemon.py /root/Workspace/xy/DiT/assets/results/ctrl_fame_1pix_v1 > /tmp/1pix_daemon.log 2>&1"
