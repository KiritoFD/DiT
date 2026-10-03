"""条件(std 图)与目标(真迹图)是否**同一个字** —— 用 32×32 墨密度相关, 不用细线 IoU。

废弃的判据: skel_iou(细线骨架, 256²) —— 对稀疏字(人/札)和笔宽差异极不鲁棒,
且空集时返回 1.0, 会产生完全错误的结论。
本判据: 两图各自 → 墨密度图(1-灰度, 32×32, 归一化) → 皮尔逊相关。
  同字(哪怕写法/角度不同): 墨团位置仍高度相关 (典型 >0.5)
  不同字 (共 vs 恭 / 繁简 異体): 相关明显更低
先用 eval200 的人眼可判样本校准阈值, 再统计训练集。
"""
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")


def ink32(p):
    im = Image.open(p).convert("L").resize((32, 32), Image.LANCZOS)
    a = 1.0 - np.asarray(im, dtype=np.float32) / 255.0
    a = a - a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 1e-6 else a


def corr(p1, p2):
    a, b = ink32(p1), ink32(p2)
    return float(np.sum(a * b))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rows = list(csv.DictReader(open("exp-std/csv/train.csv", encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    vals, per_slot, miss = [], {}, 0
    for r in rows:
        bid = os.path.basename(r["image_path"])
        ps = os.path.join("data/top10_style23/std", bid)
        pg = os.path.join("data/top10_style23/imgs", bid)
        if not (os.path.exists(ps) and os.path.exists(pg)):
            miss += 1
            continue
        c = corr(ps, pg)
        vals.append(c)
        s = r.get("slot_name")
        t = per_slot.setdefault(s, [0, 0])
        t[0] += 1
        if c < 0.30:
            t[1] += 1
    v = np.array(vals)
    print(f"[{mode}] n={len(v)} 缺文件={miss}")
    print(f"  相关分布: p5={np.percentile(v,5):.3f} p10={np.percentile(v,10):.3f} "
          f"p25={np.percentile(v,25):.3f} 中位={np.median(v):.3f} "
          f"p75={np.percentile(v,75):.3f} p90={np.percentile(v,90):.3f}")
    for t in (0.10, 0.20, 0.30, 0.40, 0.50):
        k = int((v < t).sum())
        print(f"  相关 < {t:.2f}: {k} ({k/len(v):.1%})")
    print("[按槽位] 相关<0.30 的占比 前 10:")
    for s, (n, b) in sorted(per_slot.items(), key=lambda x: -(x[1][1]/max(x[1][0],1)))[:10]:
        print(f"   {s:<14} {b:5d}/{n:<5d} = {b/max(n,1):6.1%}")


main()
