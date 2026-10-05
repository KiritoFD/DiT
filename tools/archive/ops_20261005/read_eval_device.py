import subprocess

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
grep -n -C 5 "device" /home/ds/Workspace/DiT/experiments/ablation_phase_a/eval_ablation_model.py
'"""
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== EVAL DEVICE CODE ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
