#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地标准字 skel 生成器：用 Pillow 渲染标准字(楷/隶) → 骨架图。
对比指标: 标准字骨架 vs 数据 GT 骨架的覆盖率/precision。

用法:
  python build_std_skel.py                       # 生成 _fonttest/std_skel/{kai,li}/<codepoint>.png
"""
import os, sys, csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = {
    "kai": (r"C:\Windows\Fonts\simkai.ttf", "楷体"),
    "li":  (r"C:\Windows\Fonts\SIMLI.TTF",  "隶书"),
}
SIZE = 256
OUT = os.path.join(HERE, "_fonttest", "std_skel")

def render_char(ch, font_path, size=SIZE, blank=255):
    """渲染单字到 size×size 灰度图(白底,墨=暗), 返回 uint8 数组。"""
    f = ImageFont.truetype(font_path, size=int(size*0.86))
    img = Image.new("L", (size, size), blank)
    d = ImageDraw.Draw(img)
    # 居中渲染 (用 font 的 bbox 居中)
    # 直接画, PILLOW 会把字放左上; 用 anchor 居中
    d.text((size*0.02, size*0.02), ch, font=f, fill=0)
    return np.asarray(img)

def std_skel(ch, font_path):
    """渲染标准字→二值(墨=1)→skeleton→骨架图 uint8。"""
    g = render_char(ch, font_path).astype(np.float32)
    # 归一化到 [0,255]
    lo, hi = g.min(), g.max()
    if hi - lo < 1:  # 空字
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    g = (g - lo) / max(hi - lo, 1e-6) * 255
    ink = g < 128
    # 反转: skeletonize 对亮区取骨架
    sk = skeletonize(ink)
    return (sk * 255).astype(np.uint8)

def main():
    import json
    # 读数据字集合
    chars_kai = set(open(os.path.join(HERE, "charsets", "_chars_kai.txt"), encoding="utf-8").read().strip())
    chars_li = set(open(os.path.join(HERE, "charsets", "_chars_li.txt"), encoding="utf-8").read().strip())
    # 读字体 cmap 筛可用字
    from fontTools.ttLib import TTFont
    report = {}
    for key, (fp, label) in FONTS.items():
        if not os.path.exists(fp):
            print(f"[{key}] 字体缺失 {fp}"); continue
        ft = TTFont(fp, fontNumber=0)
        cmap = set()
        for t in ft["cmap"].tables:
            if t.isUnicode(): cmap.update(t.cmap.keys())
        chars = chars_kai if key=="kai" else chars_li
        usable = sorted([c for c in chars if ord(c) in cmap])
        miss = len(chars) - len(usable)
        print(f"[{key}] {label}: 数据{len(chars)}字 → 字库可用{len(usable)} ({len(usable)/max(len(chars),1)*100:.2f}%) 缺{miss}")
        report[key] = {"usable": len(usable), "missing": miss, "total": len(chars)}
        # 渲染全部可用字
        outdir = os.path.join(OUT, key)
        os.makedirs(outdir, exist_ok=True)
        rendered = 0
        for ch in usable:
            sk = std_skel(ch, fp)
            Image.fromarray(sk).save(os.path.join(outdir, f"U+{ord(ch):05X}.png"))
            rendered += 1
            if rendered % 1000 == 0:
                print(f"   {key} ... {rendered}/{len(usable)}")
        print(f"   渲染完成 {rendered} 字 -> {outdir}")
    with open(os.path.join(HERE, "_fonttest", "std_skel_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
