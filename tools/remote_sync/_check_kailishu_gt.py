# -*- coding: utf-8 -*-
"""校验 kailishu_eval 前N行样本的 final_canny/skeleton 是否可用。"""
import csv, os

rows = list(csv.DictReader(open("kailishu_eval.csv", encoding="utf-8")))[:5]
for r in rows:
    img = r["image_path"]
    pid = os.path.basename(img)[:-4]
    c = os.path.exists(f"final_canny/{pid}.png")
    s = os.path.exists(f"final_skeleton/{pid}.png")
    print(f"{pid} | {r['script']} | {r['character']} | canny={'OK' if c else 'MISS'} skel={'OK' if s else 'MISS'}")
