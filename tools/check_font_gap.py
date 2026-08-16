#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析楷体simkai覆盖缺口: 楷书缺的字符是什么，能否用其他字体补。
并统计 '全部' 集缺口在各字体下的补充情况(多字体 union 覆盖)。"""
import os
from fontTools.ttLib import TTFont, TTCollection

CS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charsets")
FONTS = {
    "simkai": (r"C:\Windows\Fonts\simkai.ttf", "楷体"),
    "STKAITI": (r"C:\Windows\Fonts\STKAITI.TTF", "华文楷体"),
    "SIMLI": (r"C:\Windows\Fonts\SIMLI.TTF", "隶书"),
    "simsun": (r"C:\Windows\Fonts\simsun.ttc", "宋体"),
    "simhei": (r"C:\Windows\Fonts\simhei.ttf", "黑体"),
}

def load(fp):
    fonts = TTCollection(fp) if fp.lower().endswith(".ttc") else TTFont(fp, fontNumber=0)
    copies = fonts.fonts if fp.lower().endswith(".ttc") else [fonts]
    cps = set()
    for f in copies:
        if "cmap" not in f: continue
        for t in f["cmap"].tables:
            if t.isUnicode(): cps.update(t.cmap.keys())
    return cps

def main():
    all_chars = set(open(os.path.join(CS_DIR, "_chars_all.txt"), encoding="utf-8").read().strip())
    kai_chars = set(open(os.path.join(CS_DIR, "_chars_kai.txt"), encoding="utf-8").read().strip())
    maps = {}
    print("解析字体...")
    for k, (fp, label) in FONTS.items():
        if os.path.exists(fp):
            maps[k] = (load(fp), label)
    # union 覆盖
    union = set()
    for k,(cps,_) in maps.items():
        union |= cps
    cov = sum(1 for c in all_chars if ord(c) in union)
    print(f"\n多字体 union 覆盖 '全部'集(7011): {cov}/{len(all_chars)} ({cov/len(all_chars)*100:.2f}%)")
    # 楷书缺口
    kai_cps = maps["simkai"][0] if "simkai" in maps else None
    if kai_cps:
        miss = sorted(c for c in kai_chars if ord(c) not in kai_cps)
        extB = [c for c in miss if ord(c)>=0x20000]
        bmp = [c for c in miss if ord(c)<0x20000]
        def show(chs):
            return " ".join(f"U+{ord(c):04X}" for c in chs[:50])
        print(f"\n楷书 simkai 缺失 {len(miss)}: BMP {len(bmp)}, 扩展B+ {len(extB)}")
        print(f"  BMP缺失 sample: {show(bmp)}")
        print(f"  扩展B+缺 sample: {show(extB)}")
        # BMP 缺能否被 union 补
        can_fix_bmp = [c for c in bmp if ord(c) in union]
        print(f"  BMP 缺但可被其他字体补: {len(can_fix_bmp)}")
        still = [c for c in bmp if ord(c) not in union]
        print(f"  BMP 仍缺: {len(still)}")
        # 字面量(可打印)版本
        vis_extb = "".join(extB)[:60]
        print(f"  扩展B+ 字面示例: {vis_extb!r}")

if __name__ == "__main__":
    main()
