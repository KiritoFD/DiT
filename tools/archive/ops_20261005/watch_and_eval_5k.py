import os
import subprocess
import sys
import time

CKPT = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b/20261005-213114-cap_tier3_b/checkpoints/0005000.pt"
CKPT_DONE = CKPT + ".done"
OUT_DIR = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b/evaluations/eval_5k"
PYTHON = "/home/ds/miniconda3/envs/pytorch/bin/python"
EVAL_SCRIPT = (
    "/home/ds/Workspace/DiT/experiments/ablation_phase_a/eval_ablation_model.py"
)
LOG_FILE = "/home/ds/Workspace/DiT/experiments/capacity_ladder/logs/eval_5k.log"

print(f"Watching for {CKPT_DONE}...")

while True:
    if os.path.exists(CKPT_DONE) or (
        os.path.exists(CKPT) and os.path.getsize(CKPT) > 500 * 1024 * 1024
    ):
        print(f"Found {CKPT}! Waiting 5s for full disk flush...")
        time.sleep(5)
        print("Launching evaluation on 0005000.pt...")
        cmd = f"{PYTHON} -u {EVAL_SCRIPT} --ckpt {CKPT} --out-dir {OUT_DIR} > {LOG_FILE} 2>&1"
        res = subprocess.run(cmd, shell=True)
        print(f"Evaluation finished with exit code {res.returncode}!")
        break
    time.sleep(10)
