import subprocess
import sys

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
cat /home/ds/Workspace/DiT/experiments/capacity_ladder/configs/tier3_b.json
'"""
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== TIER 3 B/2 CONFIG ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
