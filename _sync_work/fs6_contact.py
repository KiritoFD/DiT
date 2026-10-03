#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fs6_contact.py — 每个主题一张拼图: 原始 wild 图 / 极性矫正后 256 图，上下两排对照。

为什么需要: 背景极性判定是逐图按"中位数<128 就反相"做的，数字上看不出一批
带底噪/灰底的拓片被"矫正"成了什么。拼图 10 秒就能否掉一个主题。
"""
import csv
import os
import sys

from PIL import Image, ImageDraw

os.chdir("/root/Workspace/xy/DiT")
TOPICS = sys.argv[1:] or ["沈周-行", "伊秉绶-行", "傅山-行", "伊秉绶-隶",
                          "宋高宗-楷", "徐渭-行"]
N, S = 8, 128
WILD = "/root/Workspace/xy/HCSU/wild_extract"
os.makedirs("_review/posters/fs6", exist_ok=True)

for t in TOPICS:
    cal, _, sc = t.partition("-")
    rows = list(csv.DictReader(open(f"assets/fs6_{t}_train.csv", encoding="utf-8")))[:N]
    sheet = Image.new("RGB", (N * S, 2 * S + 22), "white")
    dr = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        ch = r["character"]
        raw = Image.open(os.path.join(WILD, f"{cal}-{sc}", f"{ch}.png")).convert("L")
        raw.thumbnail((S, S))
        sheet.paste(raw.convert("RGB"), (i * S, 22))
        fixed = Image.open(r["image_path"]).convert("L").resize((S, S))
        sheet.paste(fixed.convert("RGB"), (i * S, S + 22))
    dr.text((4, 6), f"{t}  上: wild 原图   下: 中位数<128 反相 + 256 之后", fill="black")
    out = f"_review/posters/fs6/{t}.png"
    sheet.save(out)
    print(out)
