# -*- coding: utf-8 -*-
"""_chk_skel_bank.py v2 — 正确读取标准骨架 bank（keys 是字符串数组）。"""
import os
import csv
import numpy as np

os.chdir("/root/Workspace/xy/DiT")

for f in ["data/skel/skel_bank_std.npz", "data/skel/skel_bank_std1.npz"]:
    if not os.path.isfile(f):
        continue
    d = np.load(f, allow_pickle=True)
    print(f"\n=== {f} ===")
    print(f"  npz keys: {list(d.files)}")
    if "latents" in d and "keys" in d:
        lat, keys = d["latents"], d["keys"]
        print(f"  latents: {lat.shape} {lat.dtype}")
        print(f"  keys({len(keys)}): 例 {[str(k) for k in keys[:6]]}")
        # 解析 "楷|字" 或纯字
        chars = set()
        for k in keys:
            s = str(k)
            chars.add(s.split("|")[1] if "|" in s else s)
        print(f"  unique chars: {len(chars)}")
        # 我们的字表
        ours = set()
        for r in csv.DictReader(open("assets/train_fame_clean_v8.csv", encoding="utf-8")):
            c = r["character"]
            if len(c) == 1:
                ours.add(c)
        hit = ours & chars
        miss = ours - chars
        print(f"  我们的字表 {len(ours)} → 覆盖 {len(hit)} ({len(hit)/len(ours):.2%}), 缺 {len(miss)}")
        if 0 < len(miss) <= 60:
            print(f"    缺字: {''.join(sorted(miss))}")
        elif miss:
            print(f"    缺字样例: {''.join(sorted(miss)[:40])} ...")

# 训练 shards
for sdir in ["data/skel/std_skel1_latents_fame", "data/skel/std_skel_latents_fame"]:
    if os.path.isdir(sdir):
        fs = sorted(os.listdir(sdir))
        print(f"\n[shards] {sdir}: {len(fs)} files, 例 {fs[:3]}")
    else:
        print(f"\n[shards] {sdir}: 不存在")
