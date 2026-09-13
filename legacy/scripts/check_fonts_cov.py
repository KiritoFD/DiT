#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测本地字体覆盖哪些字符，与书法数据集需要覆盖的字做覆盖率对比。

用法:
  python check_fonts_cov.py [data_csv] [font1.ttf font2.ttc ...]
默认 data_csv 用 5script/train.csv，字体用常见的 Windows 中文字体。
输出每个字体的 coverable 字符数 + 缺哪些字（抽样显示前若干）。
"""
import csv, sys, os
from fontTools.ttLib import TTFont

def load_cmap(font_path):
    """读取字体覆盖的所有 unicode code point。支持 .ttf/.ttc/.otf"""
    try:
        fonts = None
        if font_path.lower().endswith(".ttc"):
            from fontTools.ttLib import TTCollection
            fonts = TTCollection(font_path)
        else:
            fonts = TTFont(font_path, fontNumber=0)
        # 聚合所有 face 的 cmap
        codepoints = set()
        copies = fonts.fonts if font_path.lower().endswith(".ttc") else [fonts]
        for f in copies:
            if "cmap" not in f:
                continue
            for table in f["cmap"].tables:
                if table.isUnicode():
                    for cp in table.cmap.keys():
                        codepoints.add(cp)
        return codepoints
    except Exception as e:
        print(f"  [warn] 无法解析 {font_path}: {e}")
        return None

def main():
    args = sys.argv[1:]
    data_csv = args[0] if args else "5script/train.csv"
    fonts = args[1:] if len(args) > 1 else [
        r"C:\Windows\Fonts\simkai.ttf",
        r"C:\Windows\Fonts\STKAITI.TTF",
        r"C:\Windows\Fonts\SIMLI.TTF",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    if os.path.isfile(data_csv):
        rows = list(csv.DictReader(open(data_csv, encoding="utf-8")))
        all_chars = {}
        for r in rows:
            all_chars[r["character"]] = True
        chars = set(all_chars.keys())
    else:
        chars = None

    for fp in fonts:
        if not os.path.exists(fp):
            print(f"缺失字体: {fp}")
            continue
        cp = load_cmap(fp)
        if cp is None:
            continue
        print(f"\n字体: {os.path.basename(fp)}")
        print(f"  覆盖 codepoint: {len(cp)}")
        if chars is not None:
            covered = sum(1 for ch in chars if ord(ch) in cp)
            missing = sorted(ch for ch in chars if ord(ch) not in cp)
            print(f"  覆盖数据集字: {covered}/{len(chars)} ({covered/max(len(chars),1)*100:.2f}%)")
            print(f"  缺失: {len(missing)} 字, sample: {''.join(missing[:30])}")
            # 按码区看缺失
            extB_missing = [c for c in missing if ord(c) >= 0x20000]
            bmp_missing = [c for c in missing if ord(c) < 0x20000]
            print(f"    缺失中 BMP(<0x20000): {len(bmp_missing)}, 扩展B+(>=0x20000): {len(extB_missing)}")

if __name__ == "__main__":
    main()
