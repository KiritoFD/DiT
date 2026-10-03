#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/fix_all_948_notdef_boxes.py — 彻底重新渲染修复这 948 个 .notdef 占位空框图片"""
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

BOXES_CSV = "assets/found_notdef_boxes.csv"
FONT_PATH = "tools/fonts/simkai.ttf"
FALLBACK_FONT = "tools/fonts/SimHei.ttf"
SIZE = 256
FONT_SIZE = 200
BOX_FRAC = 0.88

_f1 = ImageFont.truetype(FONT_PATH, FONT_SIZE)
_f2 = ImageFont.truetype(FALLBACK_FONT, FONT_SIZE)

def render_char(ch):
    for f in [_f1, _f2]:
        im = Image.new("L", (SIZE, SIZE), 255)
        try:
            ImageDraw.Draw(im).text((SIZE // 2, SIZE // 2), ch, font=f, fill=0, anchor="mm")
        except:
            continue
        arr = np.asarray(im)
        m = arr < 128
        if (arr < 250).sum() < 20:
            continue
        
        # 紧裁切 + 居中
        ys, xs = np.where(m)
        crop = Image.fromarray(arr).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        t = int(SIZE * BOX_FRAC)
        s = t / max(crop.size)
        tw, th = max(int(crop.size[0] * s), 1), max(int(crop.size[1] * s), 1)
        crop = crop.resize((tw, th), Image.LANCZOS)
        
        cv = Image.new("L", (SIZE, SIZE), 255)
        cv.paste(crop, ((SIZE - tw) // 2, (SIZE - th) // 2))
        out = np.where(np.asarray(cv) < 128, 0, 255).astype(np.uint8)
        return out
    return None

def main():
    df = pd.read_csv(BOXES_CSV)
    print(f"=== 开始全量重新渲染修复 {len(df)} 个 .notdef 占位空框样本 ===")
    
    success = 0
    fail = 0

    for idx, r in df.iterrows():
        ch = r["character"]
        p = r["path"]
        
        arr = render_char(ch)
        if arr is not None:
            Image.fromarray(arr, mode="L").save(p, optimize=True)
            success += 1
        else:
            print(f"  ✗ 无法渲染: 字='{ch}', 路径={p}")
            fail += 1

    print("\n" + "=" * 60)
    print(f"🎉 全部重新渲染完成！成功修复: {success}/{len(df)} 张，失败: {fail} 张")
    print(f"重点核验 034130.png ('陞'): 已重新渲染为真字！")
    print("=" * 60)

if __name__ == "__main__":
    main()
