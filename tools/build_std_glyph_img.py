#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地生成楷/隶全套标准字形渲染图(256 RGB, 白底黑字, 完整字)。
输出到 std_glyph_img/<key>/U+XXXXX.png
"""
import os, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.join(HERE, "charsets")
OUT = os.path.join(HERE, "std_glyph_img")
FONTS = {
    "kai": (r"C:\Windows\Fonts\simkai.ttf", "楷", "_chars_kai.txt"),
    "li":  (r"C:\Windows\Fonts\SIMLI.TTF",  "隶", "_chars_li.txt"),
}
SIZE = 256

def render(ch, font_path, size=SIZE):
    f = ImageFont.truetype(font_path, int(size*0.86))
    img = Image.new("RGB", (size,size), (255,255,255))
    d = ImageDraw.Draw(img)
    d.text((size*0.02, size*0.02), ch, font=f, fill=(0,0,0))
    return img

def main():
    total_ok, total_skip = 0, 0
    for key, (fp, label, csf) in FONTS.items():
        cs_path = os.path.join(CS, csf)
        if not os.path.exists(cs_path):
            print(f"[{key}] 缺字符集 {cs_path}"); continue
        chars = set(open(cs_path, encoding="utf-8").read().strip())
        # 字体 cmap
        ft = TTFont(fp, fontNumber=0)
        cmap = set()
        for t in ft["cmap"].tables:
            if t.isUnicode(): cmap.update(t.cmap.keys())
        usable = sorted(c for c in chars if ord(c) in cmap)
        out = os.path.join(OUT, key)
        os.makedirs(out, exist_ok=True)
        ok = 0
        for ch in usable:
            if os.path.exists(os.path.join(out, f"U+{ord(ch):05X}.png")):
                ok += 1; continue
            img = render(ch, fp)
            img.save(os.path.join(out, f"U+{ord(ch):05X}.png"))
            ok += 1
            if ok % 500 == 0:
                print(f"  {key} ... {ok}/{len(usable)}")
        miss = len(chars) - len(usable)
        print(f"[{key}] {label}: 数据{len(chars)}字, 字库可用{len(usable)}, 渲染{ok}, 缺字体{miss}")
        total_ok += ok; total_skip += miss
    print(f"\n完成: 渲染 {total_ok} 张, 字体缺 {total_skip} 字 (不可用)")

if __name__ == "__main__":
    main()
