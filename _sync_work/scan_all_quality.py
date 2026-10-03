"""全量扫描 GT 质量, 按书家/来源统计坏图率。

判据 (任一条成立即"坏"):
  B1 开运算保留率 < 0.50      -> 墨大部分是 1~2px 散点
  B2 最大连通块占墨比 < 0.30  -> 没有主体笔画结构
  B3 墨点率 < 0.005           -> 几乎空白
  B4 墨点率 > 0.60            -> 糊成一片
注: 开运算 = 3x3 腐蚀+膨胀, 干净图应保留 >0.9。
"""
import csv, os, sys, collections
import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label, binary_opening
from multiprocessing import Pool

os.chdir("/root/Workspace/xy/DiT")
STRUCT = np.ones((3, 3), bool)


def one(args):
    src, path = args
    try:
        g = np.asarray(Image.open(path).convert("L")
                       .resize((256, 256), Image.LANCZOS)) < 128
    except Exception:
        return (src, path, -1, -1, -1, 1)
    if not g.any():
        return (src, path, 0.0, 0.0, 0.0, 1)
    ink = g.sum()
    op = binary_opening(g, structure=STRUCT)
    keep = op.sum() / ink
    lab, nc = cc_label(g)
    sizes = np.bincount(lab.ravel())[1:]
    lcf = sizes.max() / ink if len(sizes) else 0.0
    ir = ink / g.size
    bad = int((keep < 0.50) or (lcf < 0.30) or (ir < 0.005) or (ir > 0.60))
    return (src, path, keep, lcf, ir, bad)


if __name__ == "__main__":
    o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
    w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))
    for r in o:
        r["_src"] = "old"
    for r in w:
        r["_src"] = "hcsu"
    m = o + w
    tasks = [(r["_src"], r["image_path"]) for r in m]
    print(f"扫描 {len(tasks)} 张 ...", flush=True)

    with Pool(48) as p:
        res = p.map(one, tasks, chunksize=200)

    by_c = collections.defaultdict(lambda: [0, 0])       # 书家 -> [总, 坏]
    by_src = collections.defaultdict(lambda: [0, 0])
    bad_paths = []
    for r, (src, path, keep, lcf, ir, bad) in zip(m, res):
        c = r["calligrapher"]
        by_c[c][0] += 1
        by_src[src][0] += 1
        if bad:
            by_c[c][1] += 1
            by_src[src][1] += 1
            bad_paths.append(path)

    print()
    print("=== 按来源 ===")
    for s, (t, b) in sorted(by_src.items()):
        print(f"   {s:5s} {t:>6} 张, 坏 {b:>5} = {100*b/t:5.1f}%")
    print()
    print("=== 按书家 (坏图率降序, 只列 >=1% 的) ===")
    rows = sorted(by_c.items(), key=lambda kv: -(kv[1][1] / max(kv[1][0], 1)))
    print(f"   {'书家':<10}{'总数':>7}{'坏图':>7}{'坏图率':>9}")
    for c, (t, b) in rows:
        if b / t >= 0.01:
            print(f"   {c:<10}{t:>7}{b:>7}{100*b/t:>8.1f}%")
    tot_b = sum(v[1] for v in by_c.values())
    print(f"   {'---':<10}{len(m):>7}{tot_b:>7}{100*tot_b/len(m):>8.1f}%")
    print()
    print("=== 坏图率 = 0 的书家数 ===", sum(1 for v in by_c.values() if v[1] == 0),
          "/", len(by_c))
    with open("_sync_work/bad_images.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(bad_paths))
    print(f"坏图清单 -> _sync_work/bad_images.txt ({len(bad_paths)} 条)")
