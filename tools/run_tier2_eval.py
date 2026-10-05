import os
import subprocess
import sys

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
PYTHON=/home/ds/miniconda3/envs/pytorch/bin/python
EVAL_SCRIPT=/home/ds/Workspace/DiT/experiments/ablation_phase_a/eval_ablation_model.py
CKPT_DIR=/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier2_sp/20261005-174742-cap_tier2_sp/checkpoints
OUT_BASE=/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier2_sp/evaluations

mkdir -p $OUT_BASE

echo "=== EVALUATING TIER 2 (Sp/2) @ 10,000 STEPS ==="
$PYTHON -u $EVAL_SCRIPT --ckpt $CKPT_DIR/0010000.pt --out-dir $OUT_BASE/eval_10k

echo "=== EVALUATING TIER 2 (Sp/2) @ 20,000 STEPS ==="
$PYTHON -u $EVAL_SCRIPT --ckpt $CKPT_DIR/0020000.pt --out-dir $OUT_BASE/eval_20k
'"""

res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== EVALUATION EXECUTION ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
