import os
import subprocess
import sys
import time

CKPT = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b/20261005-213114-cap_tier3_b/checkpoints/0005000.pt"
OUT_DIR = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b/evaluations/eval_5k"
PYTHON = "/home/ds/miniconda3/envs/pytorch/bin/python"
EVAL_SCRIPT = (
    "/home/ds/Workspace/DiT/experiments/ablation_phase_a/eval_ablation_model.py"
)
LOG_FILE = "/home/ds/Workspace/DiT/experiments/capacity_ladder/logs/eval_5k_cpu.log"

print(f"Launching CPU evaluation on {CKPT}...")
cmd = f"CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 {PYTHON} -u {EVAL_SCRIPT} --ckpt {CKPT} --out-dir {OUT_DIR} > {LOG_FILE} 2>&1 &"
subprocess.run(cmd, shell=True)
print("CPU evaluation launched in background!")
