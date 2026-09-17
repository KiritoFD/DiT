# -*- coding: utf-8 -*-
"""verify_std_px60.py — 校验 fame-kxl-tj-px60 的规模与"标准字"正确性.

检查项:
  1. 规模: 书法家 / 字符 / (script,char) / 分层
  2. 标准字健康度: 空白(tofu) / 过墨 / ink 分布 / 与图同尺寸
  3. 目视: (a) std 带字标签的 montage; (b) img|std 成对 montage; (c) 同一字跨书体
"""
import csv
import os
import random
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAME = "fame-kxl-tj-px60"
CSV = f"assets/train_{NAME}.csv"
OUT = f"/root/Workspace/xy/DiT/_otout_{NAME}_std"
FONT = "tools/fonts/simhei.ttf"


def lab_montage(rows, out, cols=8, cell=120, lab_h=30):
    n = min(len(rows), cols * 5)
    rr = max((n + cols - 1) // cols, 1)
    cv = Image.new("RGB", (cols * cell, rr * (cell + lab_h)), (255, 255, 255))
    d = ImageDraw.Draw(cv)
    from PIL import ImageFont
    f = ImageFont.truetype(FONT, int(lab_h * 0.6))
    for i, r in enumerate(rows[:n]):
        x, y = (i % cols) * cell, (i // cols) * (cell + lab_h)
        cv.paste(Image.open(r["std_path"]).convert("L").resize((cell, cell)).convert("RGB"), (x, y))
        d.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(200, 200, 200))
        txt = f'{r["character"]} {r["script"]}'
        d.text((x + 4, y + cell + 2), txt, fill=(0, 0, 0), font=f)
    cv.save(out)
    print(f"  -> {out}")


def pair_montage(rows, out, cell=190, gap=6):
    n = min(len(rows), 12)
    W = cell * 2 + gap * 3
    cv = Image.new("RGB", (W, n // 2 * (cell + gap) + gap), (230, 230, 230))
    for i, r in enumerate(rows[:n]):
        gx, gy = (i % 2) * (cell * 2 + gap * 2) + gap, (i // 2) * (cell + gap) + gap
        cv.paste(Image.open(r["image_path"]).convert("L").resize((cell, cell)).convert("RGB"), (gx, gy))
        cv.paste(Image.open(r["std_path"]).convert("L").resize((cell, cell)).convert("RGB"), (gx + cell + gap, gy))
    cv.save(out)
    print(f"  -> {out}")


def cross_script(rows, out, cell=170):
    by = {}
    for r in rows:
        by.setdefault(r["character"], {})[r["script"]] = r
    picks = [c for c, d in by.items() if len(d) >= 3][:8]
    scs = ["楷", "行", "隶"]
    cv = Image.new("RGB", (len(scs) * cell + 120, len(picks) * cell), (255, 255, 255))
    d = ImageDraw.Draw(cv)
    from PIL import ImageFont
    f = ImageFont.truetype(FONT, 34)
    for j, s in enumerate(scs):
        d.text((120 + j * cell + 8, 4), s, fill=(0, 0, 0), font=f)
    for i, c in enumerate(picks):
        d.text((6, i * cell + cell // 2), c, fill=(0, 0, 0), font=f)
        for j, s in enumerate(scs):
            r = by[c].get(s)
            if r:
                cv.paste(Image.open(r["std_path"]).convert("L").resize((cell, cell)).convert("RGB"),
                         (120 + j * cell, i * cell))
    cv.save(out)
    print(f"  -> {out}")
    return picks


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    cal = {r["calligrapher"] for r in rows}
    chs = {r["character"] for r in rows}
    scr = {r["script"] for r in rows}
    sc = {(r["script"], r["character"]) for r in rows}
    print(f"[{NAME}] 样本 {len(rows)}")
    print(f"  书法家 {len(cal)}   字符 {len(chs)}   书体 {len(scr)}{sorted(scr)}")
    print(f"  (script,char) 对 {len(sc)}")
    print(f"  书体分布 {dict(Counter(r['script'] for r in rows))}")
    print(f"  来源分布 {dict(Counter(r['source'] for r in rows))}")

    # 字符合法性
    bad = [c for c in chs if len(c) != 1 or not (0x3400 <= ord(c) <= 0x9FFF)]
    print(f"  非常规字符(非单字/非CJK): {len(bad)} {bad[:10]}")

    # 标准字健康度
    print("\n  === 标准字健康度 ===")
    sample = random.Random(0).sample(rows, min(4000, len(rows)))
    inks, shapes, blank, dup = [], Counter(), 0, 0
    for r in sample:
        a = np.asarray(Image.open(r["std_path"]).convert("L"))
        shapes[a.shape] += 1
        v = float((a < 128).mean())
        inks.append(v)
        if v < 0.005:
            blank += 1
    inks = np.array(inks)
    print(f"  尺寸 {dict(shapes)}")
    print(f"  ink: p1={np.percentile(inks,1):.4f} p50={np.percentile(inks,50):.3f} "
          f"p99={np.percentile(inks,99):.3f} max={inks.max():.3f}")
    print(f"  空白(ink<0.005) {blank}  |  过墨(ink>0.6) {int((inks>0.6).sum())}")

    # (script,char) 唯一性 -> 标准字应逐字唯一
    key_img = {}
    for r in rows[:20000]:
        key_img.setdefault((r["script"], r["character"]), set()).add(r["std_path"])
    multi = sum(1 for v in key_img.values() if len(v) > 1)
    print(f"  同 (script,char) 映射到多个 std 文件的: {multi} (应为 0)")

    os.makedirs(OUT, exist_ok=True)
    random.Random(1).shuffle(rows)
    lab_montage(rows, f"{OUT}/std_labeled.png")
    pair_montage(rows, f"{OUT}/img_std_pair.png")
    picks = cross_script(rows, f"{OUT}/cross_script.png")
    print(f"\n  跨书体抽检字: {picks}")

    # 列出书法家
    print(f"\n  书法家清单({len(cal)}): {' '.join(sorted(cal))}")


if __name__ == "__main__":
    main()
