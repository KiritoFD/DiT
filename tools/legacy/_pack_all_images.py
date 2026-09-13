#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""远程: 打包 cleaned CSV 中所有图片到 tar.gz"""
import csv, os, tarfile, sys

REMOTE_BASE = "/root/Workspace/xy/DiT"
CSV_PATH = os.path.join(REMOTE_BASE, "5script", "train_top30_clean.csv")
OUT_TAR = "/tmp/all_train_images.tar.gz"

# 读取所有 image_path
paths = set()
with open(CSV_PATH, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        paths.add(r["image_path"])

print(f"Total unique images in clean CSV: {len(paths)}")

# 打包
missing = 0
found = 0
with tarfile.open(OUT_TAR, "w:gz") as tar:
    for p in sorted(paths):
        full = os.path.join(REMOTE_BASE, p)
        if os.path.isfile(full):
            tar.add(full, arcname=os.path.basename(p))
            found += 1
        else:
            missing += 1
        if found % 10000 == 0:
            print(f"  packed {found}...", flush=True)

print(f"Done: {found} found, {missing} missing")
print(f"tar size: {os.path.getsize(OUT_TAR)/1e9:.2f} GB")
