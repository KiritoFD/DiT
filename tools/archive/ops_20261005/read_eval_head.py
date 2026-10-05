import subprocess

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
head -n 45 /home/ds/Workspace/DiT/experiments/ablation_phase_a/eval_ablation_model.py
'"""
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== EVAL SCRIPT HEAD ===")
print(res.stdout)
