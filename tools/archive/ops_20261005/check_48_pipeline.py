import subprocess

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
ls -la /home/ds/Workspace/DiT/experiments/capacity_ladder/logs/
cat /home/ds/Workspace/DiT/experiments/capacity_ladder/logs/pipeline.log 2>/dev/null || true
tail -n 25 /home/ds/Workspace/DiT/experiments/capacity_ladder/logs/tier2_sp_eval.log 2>/dev/null || true
'"""
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== LOGS ON 48 ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
