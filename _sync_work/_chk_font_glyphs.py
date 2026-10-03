# -*- coding: utf-8 -*-
"""_chk_font_glyphs.py — 测试现有字体能覆盖我们字表多少字（缺的 168 字能否渲染）。

渲染判定: 用 PIL 渲染字符，若墨量 < 阈值则判为缺字（tofu/空白）。
"""
import os
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")

FONTS = {
    "simkai (楷体GB2312)": "tools/fonts/simkai.ttf",
    "SIMLI (隶书)": "tools/fonts/SIMLI.TTF",
    "STXINGKA (行楷)": "tools/fonts/STXINGKA.TTF",
    "simhei (黑体)": "tools/fonts/simhei.ttf",
}

ours = set()
for r in csv.DictReader(open("assets/train_fame_clean_v8.csv", encoding="utf-8")):
    c = r["character"]
    if len(c) == 1:
        ours.add(c)
ours = sorted(ours)
print(f"我们的字表: {len(ours)} 字")


def can_render(font_path, ch, size=180):
    try:
        f = ImageFont.truetype(font_path, size)
    except Exception as e:
        return False, f"load-err {e}"
    img = Image.new("L", (size + 40, size + 40), 255)
    d = ImageDraw.Draw(img)
    d.text((20, 20), ch, font=f, fill=0)
    a = np.asarray(img)
    ink = int((a < 128).sum())
    return ink > 200, ink


cache = {}
for name, p in FONTS.items():
    if not os.path.isfile(p):
        print(f"{name}: 文件缺失 ({p})")
        continue
    ok, miss = [], []
    for ch in ours:
        r, _ = can_render(p, ch)
        (ok if r else miss).append(ch)
    print(f"\n{name}:")
    print(f"  可渲染 {len(ok)}/{len(ours)} = {len(ok)/len(ours):.2%}   缺 {len(miss)}")
    cache[name] = set(ok)
    if miss and len(miss) <= 50:
        print(f"  缺字: {''.join(miss)}")
    elif miss:
        print(f"  缺字样例: {''.join(miss[:50])} ...")

# 并集覆盖
if cache:
    uni = set().union(*cache.values())
    miss = set(ours) - uni
    print(f"\n[字体并集] 可覆盖 {len(ours)-len(miss)}/{len(ours)} "
          f"= {(len(ours)-len(miss))/len(ours):.2%}")
    if miss:
        print(f"  并集仍缺 {len(miss)} 字", end="")
        if len(miss) <= 60:
            print(": " + "".join(sorted(miss)))
        else:
            print(f" 样例: {''.join(sorted(miss)[:50])}")
        print("  → 这些字需要新字库（如 NotoSerifSC / 全字库正楷）")
    else:
        print("  → **完全覆盖，无需下载新字库**")
