#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fs6_vs.py — 同一批留出字的四联对比: 骨架 g | GT(真迹) | 不训练(直写行) | 训练 1000 步。

为什么必须看肉眼: ssim 主要由"字对不对"决定，而"字对不对"是骨架条件 g 的功劳；
风格像不像 沈周，ssim 只贡献很小一部分。判"学没学会风格"必须并排看。
"""
import csv
import glob
import os
import sys

from PIL import Image, ImageDraw

os.chdir("/root/Workspace/xy/DiT")
T = sys.argv[1] if len(sys.argv) > 1 else "沈周-行"
TRAIN_GLOBS = sys.argv[2] if len(sys.argv) > 2 else f"assets/results/v15_fs6_{T}_row_pt_lr0.001*"
S = 160
N = 12


def newest(d, step_pat):
    c = sorted(glob.glob(os.path.join(d, "**", step_pat, "fewshot", "g*.png"), recursive=True))
    return os.path.dirname(os.path.dirname(c[-1])) if c else None


rows = list(csv.DictReader(open(f"assets/fs6_{T}_eval.csv", encoding="utf-8")))[:N]
runs = sorted(glob.glob(TRAIN_GLOBS))
tr_dir = newest(runs[-1], "step*") if runs else None
ba_dir = newest(f"/tmp/_fs6_{T}_base_row_pt", "step*")
print("trained:", tr_dir, "\nbaseline:", ba_dir)

sheet = Image.new("RGB", (N * S, 4 * S + 26), "white")
dr = ImageDraw.Draw(sheet)
for i, r in enumerate(rows):
    def put(row, path):
        if path and os.path.isfile(path):
            sheet.paste(Image.open(path).convert("L").resize((S, S)).convert("RGB"),
                        (i * S, row * S + 22))
    put(0, r["std_path"])
    put(1, r["image_path"])
    put(2, os.path.join(ba_dir, "fewshot", f"g{i}.png") if ba_dir else "")
    put(3, os.path.join(tr_dir, "fewshot", f"g{i}.png") if tr_dir else "")
labels = ["骨架 g", "GT 真迹", "不训练(直写行)", "训练 1000 步 lr1e-3"]
for y, lab in enumerate(labels):
    dr.text((4, y * S + 6), lab, fill="red")
out = f"_review/posters/fs6runs/vs_{T}.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
sheet.save(out)
print(out)
