import os
import glob
import json

print("=== 1. 查找 tier3_b_aug_v66route 的评估代码 ===")
train_script = "/home/ds/Workspace/DiT/experiments/capacity_ladder"
# 查看该目录下的 .py 文件
py_files = glob.glob(f"{train_script}/*.py")
print("capacity ladder py files:", py_files)

# 查看 train 或 eval 脚本
for f in py_files:
    print(f"--- {f} ---")
    with open(f, "r", encoding="utf-8") as pf:
        lines = pf.readlines()
        for line in lines:
            if "strict" in line or "eval200" in line or "test.csv" in line:
                print(" ", line.strip())

print("\n=== 2. 查看 moyi_eval_full_metrics 是由哪个脚本生成的 ===")
moyi_base = "/home/ds/Workspace/moyi"
for py_f in glob.glob(f"{moyi_base}/*.py") + glob.glob(f"{moyi_base}/scripts/*.py"):
    with open(py_f, "r", encoding="utf-8") as pf:
        content = pf.read()
        if "eval_full_metrics" in content or "moyi_4_80k" in content:
            print("  found moyi script:", py_f)
