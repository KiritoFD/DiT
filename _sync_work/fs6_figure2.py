#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fs6_figure2.py — 每行自洽的格子数据导出（标签在本地加，远端没有中文字体）。

一行一个书家, 每行 2 个字（该主题 eval 100 字里训练后 ssim 的中位一张 + 最高一张）:
  该书家真迹 | 这个字的标准骨架(=g 输入) | 零梯度直写 | 训练 2000 步
产出: /tmp/_fs6fig2.tgz (cells/*.png + manifest.json)
"""
import csv
import glob
import json
import os

from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
TOPICS = ["沈周-行", "伊秉绶-行", "傅山-行", "伊秉绶-隶", "徐渭-行"]
CELL = 150
WORK = "/tmp/_fs6fig2"
os.makedirs(f"{WORK}/cells", exist_ok=True)

COLS = ["该书家真迹(GT)", "标准骨架(=g输入)", "零梯度直写(0步)", "训练2000步(lr1e-3)"]
manifest = {"cols": COLS, "rows": []}


def locate(pattern):
    imgs = sorted(glob.glob(os.path.join(pattern, "**", "step*", "fewshot", "g*.png"),
                            recursive=True))
    csvs = sorted(glob.glob(os.path.join(pattern, "**", "eval_stdskel_batch.csv"),
                            recursive=True), key=os.path.getmtime)
    if not imgs or not csvs:
        return None, None
    return os.path.dirname(os.path.dirname(imgs[-1])), csvs[-1]


for t in TOPICS:
    tr_dir, tr_csv = locate(f"assets/results/v15_fs6_{t}_row_pt_lr0.001*")
    ba_dir, ba_csv = locate(f"/tmp/_fs6_{t}_base_row_pt")
    ev = {r["character"]: r for r in csv.DictReader(
        open(f"assets/fs6_{t}_eval.csv", encoding="utf-8"))}
    tr = {int(r["idx"]): r for r in csv.DictReader(open(tr_csv, encoding="utf-8"))
          if r["set"] == "fewshot"}
    ba = {int(r["idx"]): r for r in csv.DictReader(open(ba_csv, encoding="utf-8"))
          if r["set"] == "fewshot"}
    scored = sorted((float(r["ssim"]), k) for k, r in tr.items()
                    if r["char"] in ev and k in ba)
    picks = [("中位", scored[len(scored) // 2][1]), ("最好", scored[-1][1])]
    for tag, idx in picks:
        r = ev[tr[idx]["char"]]
        cells = {
            "GT": r["image_path"],
            "骨架": r["std_path"],
            "直写": os.path.join(ba_dir, "fewshot", f"g{idx}.png"),
            "训练": os.path.join(tr_dir, "fewshot", f"g{idx}.png"),
        }
        entry = {"topic": t, "char": tr[idx]["char"],
                 "ssim": float(tr[idx]["ssim"]), "tag": tag, "cells": {}}
        for col, p in cells.items():
            out = f"{WORK}/cells/{t}_{tag}_{col}.png"
            im = Image.new("RGB", (CELL, CELL), (230, 230, 230))
            if os.path.isfile(p):
                im.paste(Image.open(p).convert("L").resize(
                    (CELL, CELL), Image.LANCZOS).convert("RGB"), (0, 0))
            im.save(out)
            entry["cells"][col] = os.path.basename(out)
        manifest["rows"].append(entry)

json.dump(manifest, open(f"{WORK}/manifest.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
os.system(f"cd /tmp && tar czf _fs6fig2.tgz _fs6fig2 && echo TARBALL_OK")
print("cells:", len(manifest["rows"]) * 4)
