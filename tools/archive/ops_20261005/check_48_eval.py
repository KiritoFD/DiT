import subprocess

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
ls -la /home/ds/Workspace/DiT/tools/eval_ablation_model.py
ls -la /home/ds/Workspace/DiT/experiments/capacity_ladder/
'"""
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== EVAL CHECK ON 48 ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
