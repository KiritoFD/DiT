# -*- coding: utf-8 -*-
"""check_fonts.py — 盘点本地可用中文字体, 报告各书体 zero-shot 覆盖能力.

检查:
  * Windows 系统字体目录里有哪些中文字体 (楷/行/隶/草/篆)
  * 每个字体对 fame 训练字表的实际覆盖率 (能否渲染出字)
用法:
  python tools/check_fonts.py
"""
import os
import sys
import csv
import glob
import collections

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
FONT_DIR = r"C:\Windows\Fonts"

SCRIPT_ID = {"楷": 0, "行": 3, "隶": 4, "草": 2, "篆": 1, "六体": 5}
# 期望: 每种书体用什么字体
SCRIPT_FONT = {"楷": "simkai.ttf", "行": "STXINGKA.TTF", "隶": "SIMLI.TTF",
               "草": None, "篆": None, "六体": None}

# 候选: 常见中文字体文件名 -> 猜测书体
CANDIDATES = [
    ("simkai.ttf", "楷"), ("STXINGKA.TTF", "行"), ("SIMLI.TTF", "隶"),
    ("simsun.ttc", "宋"), ("simhei.ttf", "黑"), ("msyh.ttc", "黑"),
    ("STSONG.TTF", "宋"), ("STCAIYUN.TTF", "装饰"), ("STHUPO.TTF", "装饰"),
    ("FZSTK.TTF", "楷"), ("FZYTK.TTF", "楷"), ("FZLSK.TTF", "隶"),
    ("STXINWEI.TTF", "行"), ("STLITI.TTF", "隶"), ("STFANGSO.TTF", "仿"),
]


def probe(path, chars, size=200):
    """用字体渲染 chars, 返回 (能渲染的字数, 总数)."""
    try:
        f = ImageFont.truetype(path, size)
    except Exception:
        return 0, len(chars), "load-fail"
    ok, miss = 0, 0
    for ch in chars:
        img = Image.new("L", (256, 256), 255)
        d = ImageDraw.Draw(img)
        try:
            d.text((128, 128), ch, font=f, fill=0, anchor="mm")
        except Exception:
            miss += 1
            continue
        arr = np.asarray(img)
        if (arr < 250).sum() >= 10:
            ok += 1
        else:
            miss += 1
    return ok, len(chars), "ok"


def main():
    rows = list(csv.DictReader(open("5script/train_fame.csv", encoding="utf-8")))
    # 每种书体的唯一字
    uniq = collections.defaultdict(set)
    for r in rows:
        uniq[r["script"]].add(r["character"])

    print("=== 1. 系统字体目录中的候选字体 ===")
    found = []
    for fn, guess in CANDIDATES:
        p = os.path.join(FONT_DIR, fn)
        if os.path.isfile(p):
            found.append((fn, guess, p, os.path.getsize(p)))
    for fn, g, p, sz in found:
        print(f"  {fn:<18s} 猜测书体={g:<6s} {sz//1024:>6d} KB")
    print(f"  找到 {len(found)}/{len(CANDIDATES)}")

    print("\n=== 2. 当前 SCRIPT_FONT 配置状态 ===")
    for k in SCRIPT_ID:
        f = SCRIPT_FONT.get(k)
        if f:
            p = os.path.join(FONT_DIR, f)
            st = "OK" if os.path.isfile(p) else "缺失!"
            print(f"  {k:<6s} -> {f:<16s} [{st}]")
        else:
            print(f"  {k:<6s} -> 无专用字体  需从 {len(uniq[k])} 唯一字中 zero-shot")

    print("\n=== 3. 现有字体对各书体字表的覆盖率 (抽样 300 字) ===")
    print(f"{'书体':<6s}{'唯一字':>8s}{'用字体':>18s}{'可渲染':>9s}{'覆盖率':>9s}")
    for k in ["楷", "行", "隶", "草", "篆", "六体"]:
        chars = sorted(uniq[k])
        if not chars:
            continue
        samp = chars[::max(1, len(chars) // 300)][:300]
        f = SCRIPT_FONT.get(k) or "simkai.ttf"   # 无字体时回退楷体
        p = os.path.join(FONT_DIR, f)
        if not os.path.isfile(p):
            print(f"  {k:<6s}{len(chars):8d}{f:>18s}{'字体缺失':>9s}")
            continue
        ok, tot, st = probe(p, samp)
        print(f"  {k:<6s}{len(chars):8d}{f:>18s}{ok:6d}/{tot:<4d}"
              f"{ok/tot*100:8.1f}%")

    print("\n=== 4. 目录里所有 .ttf/.ttc (前 30) ===")
    ttf = sorted(glob.glob(os.path.join(FONT_DIR, "*.tt?")))
    print(f"  共 {len(ttf)} 个 ttf/ttc")
    for p in ttf[:30]:
        print(f"    {os.path.basename(p)}")


if __name__ == "__main__":
    main()
