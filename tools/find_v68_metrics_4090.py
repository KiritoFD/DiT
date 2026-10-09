import os
import glob
import pandas as pd
import json

base_4090 = "/root/Workspace/xy/DiT"
print("=== 1. 查找 4090 上 v68 的详细 eval 结果 ===")
# 检查 v68 目录下的 csv 或 json
v68_dir = "/root/Workspace/xy/DiT/assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot"
for f in glob.glob(f"{v68_dir}/**/*.csv", recursive=True) + glob.glob(f"{v68_dir}/**/*.json", recursive=True):
    print("  found in v68:", f)

print("\n=== 2. 检查 eval200fix 目录下是否有 per-sample 指标 ===")
for p in glob.glob(f"{v68_dir}/**/eval200fix", recursive=True):
    print("  eval200fix dir:", p)
    files = os.listdir(p)
    print("    files count:", len(files))
    non_png = [f for f in files if not f.endswith(".png")]
    print("    non_png files:", non_png)
