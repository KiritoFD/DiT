import subprocess
import sys

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 'ps -ef | grep python; ls -la /home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier2_sp/'"""
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== PS ON 48 ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
