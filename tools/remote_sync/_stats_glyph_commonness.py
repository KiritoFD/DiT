# -*- coding: utf-8 -*-
"""评估字库覆盖：把五体字按常用字程度拆分。
- GB2312 一级(3755)+二级(3008)=6763 常用字：绝大多数开源字体覆盖
- 扩展生僻字(U+3400+等)：主流字体基本不覆盖

输出各书体里"常用字"占比，判断字库能覆盖多少。
"""
import csv

# 内置一张尽量大的常见字判断：用最低原则——字符码点 < 0x3400（BMP 统一区）
# 且属于 CJK Unified（0x4E00-0x9FFF）或基本区，视为"常见"；
# 0x3400-0x4DBF 是扩展A（较生僻）。
def is_common(ch):
    cp = ord(ch)
    # 常用主流字体一般覆盖到扩展A(U+3400-U+4DBF)部分，扩展B+(U+20000+)基本都不覆盖
    if cp >= 0x20000:   # 扩展B及以上，几乎无开源字库覆盖
        return "extB"
    if 0x3400 <= cp < 0x4E00:  # 扩展A
        return "extA"
    if 0x4E00 <= cp < 0x9FFF:  # CJK 基本区（含常用与次常用）
        return "bmp"
    return "other"

from collections import defaultdict
rows = list(csv.DictReader(open("5script/train.csv", encoding="utf-8")))
chars_by_script = defaultdict(set)
for r in rows:
    chars_by_script[int(r["script_id"])].add(r["character"])

name = {0:"楷",1:"篆",2:"草",3:"行",4:"隶"}
print("=== 各书体按码位区域的字数分布 ===")
for sid in sorted(chars_by_script):
    s = chars_by_script[sid]
    dc = defaultdict(int)
    for ch in s:
        dc[is_common(ch)] += 1
    tot = len(s)
    bmp = dc.get('bmp',0); exta=dc.get('extA',0); extb=dc.get('extB',0)
    print(f"  {name[sid]}: 总{len(s)} | BMP基础区(≈字库易覆盖){bmp}({bmp/tot*100:.1f}%) "
          f"| 扩展A(部分字库){exta}({exta/tot*100:.1f}%) | 扩展B+(难覆盖){extb}({extb/tot*100:.1f}%)")
