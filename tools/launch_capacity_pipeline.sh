#!/usr/bin/env bash
cd /home/ds/Workspace/DiT
export PYTHONPATH=/home/ds/Workspace/DiT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/home/ds/miniconda3/envs/pytorch/bin/python

# Kill previous session if any exists
tmux kill-session -t capacity_ladder 2>/dev/null || true

tmux new-session -d -s capacity_ladder "${PYTHON} -u experiments/capacity_ladder/run_capacity_pipeline.py > experiments/capacity_ladder/logs/pipeline_runner.log 2>&1"

echo "Capacity Ladder Tmux session launched:"
tmux ls
