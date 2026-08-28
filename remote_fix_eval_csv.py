#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 eval/show5/seen5 CSV 的 image_path 列从 final_images/ 改成 final_imgs_256/。
final_images 已删除，只留 256 版本。在远程原地修改（s7 已结束，不影响）。"""
import csv, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "/root/Workspace/xy/DiT"
for name in ["eval100_top30_clean.csv", "show5_top30.csv", "seen5_top30.csv"]:
    p = os.path.join(BASE, "5script", name)
    if not os.path.exists(p):
        print(f"SKIP {name}: not found")
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    changed = 0
    for r in rows:
        if r["image_path"].startswith("final_images/"):
            r["image_path"] = r["image_path"].replace("final_images/", "final_imgs_256/", 1)
            changed += 1
    # 写回
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows, {changed} paths fixed -> final_imgs_256/")
    # 验证第一行文件存在
    first = os.path.join(BASE, rows[0]["image_path"])
    print(f"  check first: {rows[0]['image_path']} exists={os.path.isfile(first)}")