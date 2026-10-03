#!/bin/bash
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT:$PYTHONPATH
exec /opt/conda/bin/python src/eval/eval_metrics_daemon.py assets/results/s28_std_dino_pretrain
