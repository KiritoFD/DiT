#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断简体/繁体: 对 show5 的字, 看:
  1. 数据集 GT 的字符码点 + 字形名
  2. 标准字库(simkai kai/SIMLI li)是否能渲染该字, 渲染字形是否繁体优先
  3. 用标宋/宋体(simsun 简体)对比: 若 GT 字是繁体, simsun 简体会渲染成不同结构。
"""
import os, csv
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "show5_eval.csv"), encoding="utf-8")))[:5]

# 字体 cmap 集合
def load_cmap(fp):
    try:
        ft = TTFont(fp, fontNumber=0) if not fp.lower().endswith(".ttc") else None
        if ft is None:
            from fontTools.ttLib import TTCollection
            ft = TTCollection(fp)[0]
        cps = set()
        for t in ft["cmap"].tables:
            if t.isUnicode(): cps.update(t.cmap.keys())
        return cps
    except Exception as e:
        return None

fonts = {
    "楷simkai(简体)": r"C:\Windows\Fonts\simkai.ttf",
    "隶SIMLI": r"C:\Windows\Fonts\SIMLI.TTF",
    "宋simsun(简体)": r"C:\Windows\Fonts\simsun.ttc",
    "华文楷STKAITI": r"C:\Windows\Fonts\STKAITI.TTF",
}
cps = {name: load_cmap(p) for name, p in fonts.items() if os.path.exists(p)}

print("=== show5 字: 码点 + 各字体能否渲染 ===")
for i, r in enumerate(rows):
    ch = r["character"]
    cp = ord(ch)
    blk = "CJK基本区" if 0x4E00 <= cp < 0x9FFF else ("扩展A" if 0x3400 <= cp < 0x4E00 else "其他")
    render = {n: (cp in c) for n, c in cps.items()}
    print(f"[{i}] {ch!r} U+{cp:04X} [{blk}] script={r['script']}")
    print(f"    渲染: " + " ".join(f"{n}:{'Y' if v else 'N'}" for n, v in render.items()))
