import pandas as pd
import os

csv_path = "/root/Workspace/xy/DiT/exp-std/csv/eval200_fixed.csv"
df = pd.read_csv(csv_path)

targets = [
    (10042, "柳公权", "楷", "連"),
    (17513, "王羲之", "楷", "旨"),
    (17687, "王羲之", "行", "好"),
    (18038, "米芾",   "行", "墟"),
    (14290, "赵孟頫", "行", "匠"),
    (37234, "赵孟頫", "隶", "盤"),
    (20672, "颜真卿", "楷", "其"),
    (4219,  "颜真卿", "行", "憫"),
]

eval_dir = "/root/Workspace/xy/DiT/exp-std/runs_purestd/eval_samples_ctrl/step0022500/eval200fix"

for tid, c, s, ch in targets:
    m = df[df["img_id"] == tid]
    if len(m):
        idx = m.index[0]
        fname = f"g{idx}.png"
        fpath = os.path.join(eval_dir, fname)
        exists = os.path.exists(fpath)
        print(f"img_id={tid:5d} | {c} {s} {ch} -> idx={idx:3d} | {fname} exists={exists}")
    else:
        print(f"img_id={tid:5d} NOT FOUND in csv")
