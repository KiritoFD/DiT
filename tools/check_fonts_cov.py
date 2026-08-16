#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地检测每个字体对 5 书体字符集的覆盖率。
读 charsets/_chars_{kai,xing,cao,li,zhuan,all}.txt，与本地字体 cmap 对比。"""
import csv, sys, os
from fontTools.ttLib import TTFont, TTCollection

CHARSETS = {
    "楷": "kai", "行": "xing", "草": "cao", "隶": "li", "篆": "zhuan", "全部": "all",
}
HERE = os.path.dirname(os.path.abspath(__file__))
CS_DIR = os.path.join(HERE, "charsets")

FONTS = [
    (r"C:\Windows\Fonts\simkai.ttf", "楷体simkai"),
    (r"C:\Windows\Fonts\STKAITI.TTF", "华文楷体STKAITI"),
    (r"C:\Windows\Fonts\SIMLI.TTF", "隶书SIMLI"),
    (r"C:\Windows\Fonts\simsun.ttc", "宋体simsun"),
    (r"C:\Windows\Fonts\simhei.ttf", "黑体simhei"),
    (r"C:\Windows\Fonts\simsunb.ttf", "新宋体simsunb"),
    (r"C:\Windows\Fonts\SimsunExtG.ttf", "宋体扩展G"),
    (r"C:\Windows\Fonts\Deng.ttf", "等线Deng"),
]

def load_cmap(fp):
    try:
        fonts = TTCollection(fp) if fp.lower().endswith(".ttc") else TTFont(fp, fontNumber=0)
        copies = fonts.fonts if fp.lower().endswith(".ttc") else [fonts]
        cps = set()
        for f in copies:
            if "cmap" not in f: continue
            for t in f["cmap"].tables:
                if t.isUnicode():
                    cps.update(t.cmap.keys())
        return cps
    except Exception as e:
        print(f"  [warn] {fp}: {e}")
        return None

def main():
    sets = {}
    for zh, en in CHARSETS.items():
        p = os.path.join(CS_DIR, f"_chars_{en}.txt")
        if os.path.exists(p):
            sets[zh] = set(open(p, encoding="utf-8").read().strip())
    # 表头
    header = "字体".ljust(18) + "".join(f"{zh:>12}" for zh in sets)
    print(header)
    print("-" * len(header))
    for fp, label in FONTS:
        if not os.path.exists(fp):
            print(f"{label:<20} 缺失 {os.path.basename(fp)}")
            continue
        cps = load_cmap(fp)
        if cps is None:
            continue
        row = f"{label:<20}"
        for zh, s in sets.items():
            covered = sum(1 for c in s if ord(c) in cps)
            pct = covered / max(len(s), 1) * 100
            row += f"  {pct:>8.1f}%({covered})"
        print(row)
    print("\n注: 楷体simkai/华文楷体配楷书; 隶书配隶; 宋体/黑体作内容fallback候选.")

if __name__ == "__main__":
    main()
