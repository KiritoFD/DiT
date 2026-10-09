import os

eval_script = "/home/ds/Workspace/DiT/experiments/ablation_phase_a/eval_ablation_model.py"
print(f"=== 查看 {eval_script} ===")
with open(eval_script, "r", encoding="utf-8") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if any(k in line for k in ["csv", "strict", "save", "png", "eval", "sample", "indices"]):
        print(f"[{i+1}] {line.strip()}")
