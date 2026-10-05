import subprocess

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
ls -la /home/ds/Workspace/DiT/experiments/ablation_phase_a/eval_ablation_model.py 2>/dev/null || echo "NOT FOUND"
ls -la /home/ds/Workspace/DiT/experiments/ablation_phase_a/ 2>/dev/null || echo "DIR NOT FOUND"
'"""
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== EVAL SCRIPT CHECK ===")
print(res.stdout)
