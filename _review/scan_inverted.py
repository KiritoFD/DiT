"""扫反色图 (白字黑底)。

判据: 四角 + 四边中点的灰度。正常图背景是白(255)，反色图背景是黑(0)。
另报 ink>0.45 的"墨重"图 (隶书/漆书本来就粗, 需人工区分)。
"""
import csv, os, collections
import numpy as np
from PIL import Image
from multiprocessing import Pool

os.chdir("/root/Workspace/xy/DiT")
rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))


def one(p):
    try:
        a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    except Exception:
        return (0.0, 0.0, -1)
    h, w = a.shape
    m = max(4, h // 32)
    edge = np.concatenate([a[:m].ravel(), a[-m:].ravel(),
                           a[:, :m].ravel(), a[:, -m:].ravel()])
    bg = float(edge.mean()) / 255.0          # 边框平均亮度, 正常图应接近 1
    ink = float((a < 128).mean())
    return (bg, ink, 0)


if __name__ == "__main__":
    with Pool(48) as pl:
        res = pl.map(one, [r["image_path"] for r in rows], chunksize=200)
    bg = np.array([r[0] for r in res])
    ink = np.array([r[1] for r in res])
    print(f"边框亮度: med={np.median(bg):.3f} p1={np.percentile(bg,1):.3f} "
          f"min={bg.min():.3f}")
    print(f"墨点率  : med={np.median(ink):.3f} p99={np.percentile(ink,99):.3f} "
          f"max={ink.max():.3f}")
    print()
    for t in [0.5, 0.3, 0.15, 0.05]:
        n = int((bg < t).sum())
        print(f"  边框亮度 < {t:.2f} (越暗越像反色): {n:>5} ({100*n/len(rows):.3f}%)")
    inv = np.where(bg < 0.30)[0]
    print()
    print("疑似反色图的来源分布:",
          collections.Counter(rows[i]["source"] for i in inv).most_common(5))
    print("疑似反色图的书家 top8:",
          collections.Counter(rows[i]["calligrapher"] for i in inv).most_common(8))
    with open("_sync_work/inverted.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rows[i]["image_path"] for i in inv))
    print(f"-> _sync_work/inverted.txt ({len(inv)} 条)")
    # 墨重但非反色
    heavy = np.where((ink > 0.45) & (bg >= 0.30))[0]
    print()
    print(f"墨重但非反色 (ink>0.45, 边框正常): {len(heavy)} 张")
    print("  书家 top8:", collections.Counter(rows[i]["calligrapher"]
                                              for i in heavy).most_common(8))
    with open("_sync_work/heavy_ink_ok.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rows[i]["image_path"] for i in heavy))
