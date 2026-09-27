#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd

ROOT = "/root/Workspace/xy/DiT"
df = pd.read_csv(os.path.join(ROOT, "assets/train_top10_style23.csv"))

missing_img = []
missing_std = []

for idx, r in df.iterrows():
    p_img = os.path.join(ROOT, r["image_path"])
    p_std = os.path.join(ROOT, r["std_path"])
    if not os.path.exists(p_img):
        missing_img.append((idx, r["image_path"]))
    if not os.path.exists(p_std):
        missing_std.append((idx, r["std_path"]))

print(f"Total rows: {len(df)}")
print(f"Missing images: {len(missing_img)}")
print(f"Missing std: {len(missing_std)}")
if missing_img:
    print("First 3 missing img:", missing_img[:3])
if missing_std:
    print("First 3 missing std:", missing_std[:3])
