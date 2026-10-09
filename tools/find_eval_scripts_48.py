import os
import glob

print("=== 1. 查找包含 eval_full_metrics 的脚本 ===")
for root, dirs, files in os.walk("/home/ds/Workspace"):
    if ".git" in root or "checkpoints" in root:
        continue
    for f in files:
        if f.endswith(".py") or f.endswith(".sh"):
            fp = os.path.join(root, f)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fobj:
                    txt = fobj.read()
                    if "eval_full_metrics" in txt or "moyi_4_80k" in txt or "moyi_12_50k" in txt:
                        print("  found:", fp)
            except Exception:
                pass

print("\n=== 2. 查看 run_capacity_pipeline.py 里的评估调用 ===")
with open("/home/ds/Workspace/DiT/experiments/capacity_ladder/run_capacity_pipeline.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "eval" in line or "python" in line or "test.csv" in line:
            print(f"  [{i+1}] {line.strip()}")
