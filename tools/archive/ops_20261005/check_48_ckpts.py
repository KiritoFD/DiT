import subprocess
import sys

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
ls -lh /home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier2_sp/20261005-174742-cap_tier2_sp/checkpoints/
'"""
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== CKPTS ON 48 ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
