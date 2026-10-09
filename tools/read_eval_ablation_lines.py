import os

with open("/home/ds/Workspace/DiT/experiments/ablation_phase_a/eval_ablation_model.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines[70:145], start=71):
    print(f"[{i:03d}] {l.rstrip()}")
