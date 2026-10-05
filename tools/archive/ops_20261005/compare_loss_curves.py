import subprocess
import sys

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
PYTHON=/home/ds/miniconda3/envs/pytorch/bin/python
$PYTHON -c "
import re

def parse_log(path):
    steps = {}
    with open(path) as f:
        for line in f:
            m = re.search(r\"step=(\\d+).*?Diff:\\s*([0-9\\.]+)\", line)
            if m:
                st = int(m.group(1))
                diff = float(m.group(2))
                steps[st] = diff
    return steps

sp_log = \"/home/ds/Workspace/DiT/experiments/capacity_ladder/logs/tier2_sp_train.log\"
b_log = \"/home/ds/Workspace/DiT/experiments/capacity_ladder/logs/tier3_b_train.log\"

sp_data = parse_log(sp_log)
b_data = parse_log(b_log)

print(f\"{'Step':<8} | {'Tier 2 (Sp/2, 59M) Loss':<24} | {'Tier 3 (B/2, 131M) Loss':<24} | {'Delta':<10}\")
print('-' * 75)
for s in sorted(b_data.keys()):
    if s in sp_data and s % 500 == 0:
        delta = b_data[s] - sp_data[s]
        sign = '+' if delta > 0 else ''
        print(f\"{s:<8d} | {sp_data[s]:<24.4f} | {b_data[s]:<24.4f} | {sign}{delta:<10.4f}\")
"
'"""

res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== LOSS CURVE COMPARISON ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
