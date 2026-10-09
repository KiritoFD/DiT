import os
import glob
import json
import csv

print("=== 1. moyi_top10_rf (12ch) 目录 ===")
p12 = "/home/ds/Workspace/moyi/results/moyi_top10_rf"
if os.path.exists(p12):
    for sub in sorted(os.listdir(p12)):
        sub_p = os.path.join(p12, sub)
        if os.path.isdir(sub_p):
            n_png = len(glob.glob(sub_p + "/*.png"))
            print(f"  {sub}: {n_png} 张 PNG")
            pngs = sorted([os.path.basename(f) for f in glob.glob(sub_p + "/*.png")])[:4]
            print(f"    示例: {pngs}")

print("\n=== 2. moyi_top10_rf_4ch (4ch) 目录 ===")
p4 = "/home/ds/Workspace/moyi/results/moyi_top10_rf_4ch"
if os.path.exists(p4):
    for sub in sorted(os.listdir(p4)):
        sub_p = os.path.join(p4, sub)
        if os.path.isdir(sub_p):
            n_png = len(glob.glob(sub_p + "/*.png"))
            print(f"  {sub}: {n_png} 张 PNG")
            pngs = sorted([os.path.basename(f) for f in glob.glob(sub_p + "/*.png")])[:4]
            print(f"    示例: {pngs}")

print("\n=== 3. eval_full_metrics 指标内容 ===")
pm = "/home/ds/Workspace/moyi/results/eval_full_metrics"
if os.path.exists(pm):
    for f in sorted(os.listdir(pm)):
        fp = os.path.join(pm, f)
        print(f"  {f} ({os.path.getsize(fp)} bytes):")
        if f.endswith(".json"):
            try:
                with open(fp, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    print("    ", str(data)[:200])
            except Exception as e:
                print("    err:", e)
        elif f.endswith(".csv"):
            try:
                with open(fp, "r", encoding="utf-8") as cf:
                    reader = csv.reader(cf)
                    header = next(reader, None)
                    row1 = next(reader, None)
                    print(f"     header: {header[:8]}")
                    print(f"     row1: {row1[:8] if row1 else 'None'}")
            except Exception as e:
                print("     err:", e)

print("\n=== 4. 48 DiT capacity ladder 与 SOTA 实验评测 ===")
pdit = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results"
if os.path.exists(pdit):
    for exp in sorted(os.listdir(pdit)):
        evals = glob.glob(f"{pdit}/{exp}/**/eval*", recursive=True)
        print(f"  {exp}: {len(evals)} 个 eval 目录")
        for ev in evals[:2]:
            n_png = len(glob.glob(ev + "/*.png"))
            print(f"    {os.path.basename(ev)}: {n_png} 张 PNG")

print("\n=== 5. 48 DiT results 其它实验 ===")
pdit2 = "/home/ds/Workspace/DiT/results"
if os.path.exists(pdit2):
    for exp in sorted(os.listdir(pdit2)):
        exp_p = os.path.join(pdit2, exp)
        if os.path.isdir(exp_p):
            evals = glob.glob(f"{exp_p}/**/eval*", recursive=True)
            print(f"  {exp}: {len(evals)} 个 eval 目录")
