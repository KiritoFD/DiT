import os
import csv
import json

print("=== 1. 查找 eval_full_metrics 的元数据 ===")
# 检查 /home/ds/Workspace/moyi 里的 csv
for p in ["/home/ds/Workspace/moyi/exp-std-csv/eval200_fixed.csv", "/home/ds/Workspace/moyi/results/eval_full_metrics"]:
    if os.path.exists(p):
        print("  found:", p)

# 查看 eval_full_metrics 目录下是否有其它文本文件
p_m = "/home/ds/Workspace/moyi/results/eval_full_metrics"
for f in os.listdir(p_m):
    if f.endswith(".txt") or f.endswith(".json") or f.endswith(".csv"):
        print("  metrics file:", f)
        with open(os.path.join(p_m, f), "r", encoding="utf-8") as jf:
            print("   content:", jf.read()[:200])

print("\n=== 2. 查看 tier3_b_aug_v66route 的 eval_60000 ===")
p_b = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug_v66route/20261006-202826-v66_cap_tier3_b_aug_route2456/evaluations/eval_60000"
if os.path.exists(p_b):
    print("  tier3_b_aug_v66route eval_60000 files:")
    print(" ", sorted(os.listdir(p_b)))

print("\n=== 3. 查看 moyi_top10_rf/eval_ours200fix ===")
p_200 = "/home/ds/Workspace/moyi/results/moyi_top10_rf/eval_ours200fix"
if os.path.exists(p_200):
    files = sorted(os.listdir(p_200))
    print(f"  eval_ours200fix: {len(files)} files")
    # 查找是否有 csv 或 json
    meta = [f for f in files if not f.endswith(".png")]
    print(f"  meta files: {meta}")
    # 采样看前几个 g
    print(f"  samples: {[f for f in files if f.startswith('g')][:10]}")
