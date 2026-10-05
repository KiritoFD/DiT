#!/usr/bin/env bash
cd /home/ds/Workspace/DiT
export PYTHONPATH=/home/ds/Workspace/DiT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/home/ds/miniconda3/envs/pytorch/bin/python

# Kill previous session if any exists
tmux kill-session -t ablation_phase_a 2>/dev/null || true

tmux new-session -d -s ablation_phase_a "${PYTHON} -u experiments/ablation_phase_a/run_pipeline_phase_a.py > experiments/ablation_phase_a/logs/pipeline_runner.log 2>&1"

echo "Tmux session launched:"
tmux ls
