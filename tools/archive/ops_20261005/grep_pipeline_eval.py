import subprocess

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
grep -n "eval" /home/ds/Workspace/DiT/experiments/capacity_ladder/run_capacity_pipeline.py | head -n 30
'"""
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== GREP EVAL IN RUN_CAPACITY_PIPELINE.PY ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
