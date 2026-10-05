import subprocess
import sys
import time

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
mkdir -p /home/ds/Workspace/DiT/experiments/capacity_ladder/logs
mkdir -p /home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b

cd /home/ds/Workspace/DiT

PYTHON=/home/ds/miniconda3/envs/pytorch/bin/python
CONFIG=/home/ds/Workspace/DiT/experiments/capacity_ladder/configs/tier3_b.json
LOG=/home/ds/Workspace/DiT/experiments/capacity_ladder/logs/tier3_b_train.log

echo "Launching Tier 3 (B/2, 131M params, Batch 384) in background with PYTHONPATH..."
export PYTHONPATH=/home/ds/Workspace/DiT
nohup $PYTHON -u src/train/train.py --config $CONFIG > $LOG 2>&1 &
sleep 4
ps -ef | grep train.py | grep -v grep
'"""

res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== LAUNCH OUTPUT ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
