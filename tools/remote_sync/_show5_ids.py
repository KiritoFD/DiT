# -*- coding: utf-8 -*-
"""固定 show5 样本：列出 eval_csv 前5行 id 及其 final_canny/skeleton GT 是否存在。"""
import csv, os

csvf = "5script/eval_strata/clean_unseen_triple_100.csv"
rows = list(csv.DictReader(open(csvf, encoding="utf-8")))[:5]
print("show5 固定样本 (id, callig, script, char):")
for r in rows:
    img = r["image_path"]
    pid = os.path.basename(img)[:-4]
    c = os.path.exists(f"final_canny/{pid}.png")
    s = os.path.exists(f"final_skeleton/{pid}.png")
    print(f"  {pid} | {r['calligrapher']} | {r['script']} | {r['character']} | canny_gt={'OK' if c else 'MISS'} skel_gt={'OK' if s else 'MISS'}")
