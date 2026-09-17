# -*- coding: utf-8 -*-
"""stat_skel_ink.py — 统计 g 条件源骨架 PNG 的墨占比, 找"空白骨架".

若某些 (script,char) 的标准字骨架 PNG 是空白/近空白, 复用它的样本就拿到一个
"无内容"的 g 条件 -> 模型面对自相矛盾的监督, 输出糊团/黑块且 loss 压不下去。
"""
import csv
import glob
import multiprocessing as mp
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PNG_DIR = "data/skel/std_skel3_base_png"
KEY2UID = "data/skel/std_skel3_base_key2uid.csv"


def ink(p):
    try:
        a = np.asarray(Image.open(p).convert("L"))
        return (p, float((a < 128).mean()))
    except Exception:
        return (p, -1.0)


def main():
    files = sorted(glob.glob(os.path.join(PNG_DIR, "*.png")))
    print(f"[stat] {PNG_DIR}: {len(files)} 张")
    with mp.Pool(40) as pool:
        res = pool.map(ink, files, chunksize=128)

    inks = np.array([r[1] for r in res], dtype=np.float32)
    print(f"  ink 分布: min={inks.min():.4f} p1={np.percentile(inks,1):.4f} "
          f"p5={np.percentile(inks,5):.4f} p50={np.percentile(inks,50):.4f} "
          f"max={inks.max():.4f}")
    for t in (0.001, 0.005, 0.01, 0.02, 0.03):
        n = int((inks < t).sum())
        print(f"  ink < {t:<6} : {n:7d} ({100*n/len(inks):.2f}%)")

    blank = [(r[0], r[1]) for r in res if 0 <= r[1] < 0.01]
    print(f"\n  空白/近空白骨架样例 ({len(blank)}):")
    for p, v in blank[:15]:
        print(f"    ink={v:.5f}  {os.path.basename(p)}")

    # 映射到 (script, char) -> 受影响样本数
    if os.path.exists(KEY2UID):
        uid2key = {}
        for r in csv.DictReader(open(KEY2UID, encoding="utf-8")):
            try:
                uid2key[int(r["uid"])] = (r.get("script", ""), r.get("character", ""))
            except Exception:
                pass
        blankset = {int(os.path.basename(p)[:-4]) for p, v in blank}
        print(f"\n  空白骨架对应的 (script,char): "
              f"{[uid2key.get(u) for u in list(blankset)[:15]]}")

    # 统计训练 csv 里有多少行命中空白骨架
    for csvp in ("assets/train_base_sym_clean.csv", "assets/train_base_clean.csv"):
        if not os.path.exists(csvp):
            continue
        rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
        key2uid = {}
        for r in csv.DictReader(open(KEY2UID, encoding="utf-8")):
            key2uid[(r.get("script", ""), r.get("character", ""))] = int(r["uid"])
        n = 0
        bad_chars = Counter()
        for r in rows:
            u = key2uid.get((r.get("script", ""), r.get("character", "")))
            if u in blankset:
                n += 1
                bad_chars[(r.get("script", ""), r.get("character", ""))] += 1
        print(f"\n  {csvp}: {len(rows)} 行, 其中 {n} 行 "
              f"({100*n/max(len(rows),1):.2f}%) 命中空白骨架; "
              f"涉及 {len(bad_chars)} 个字符")
        if bad_chars:
            print(f"    Top10: {bad_chars.most_common(10)}")


if __name__ == "__main__":
    main()
