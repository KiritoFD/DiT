"""修正判据后重扫。

教训: 上一版用 "最大连通块占墨比 < 0.30" 判坏 -> **大面积误判**。
书法字(尤其楷/瘦金)笔画本来就不连通, lcf 5~30% 完全正常。
目检验证后改为只按**开运算保留率 keep** 分档:

  keep >= 0.60       干净
  0.30 <= keep < 0.60 轻微散点 (可用)
  0.05 <= keep < 0.30 重度散点 (主体笔画还在, 可考虑修)
  keep <  0.05       纯噪点 (没有笔画结构) -> **丢弃**
"""
import csv, os, collections
import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label, binary_opening
from multiprocessing import Pool

os.chdir("/root/Workspace/xy/DiT")
STRUCT = np.ones((3, 3), bool)


def one(path):
    try:
        g = np.asarray(Image.open(path).convert("L")
                       .resize((256, 256), Image.LANCZOS)) < 128
    except Exception:
        return (-1.0, -1.0, -1.0)
    if not g.any():
        return (0.0, 0.0, 0.0)
    ink = g.sum()
    keep = binary_opening(g, structure=STRUCT).sum() / ink
    lab, nc = cc_label(g)
    sizes = np.bincount(lab.ravel())[1:]
    lcf = sizes.max() / ink if len(sizes) else 0.0
    return (keep, lcf, ink / g.size)


if __name__ == "__main__":
    o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
    w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))
    for r in o:
        r["_src"] = "old"
    for r in w:
        r["_src"] = "hcsu"
    m = o + w
    print(f"扫描 {len(m)} 张 ...", flush=True)
    with Pool(48) as p:
        res = p.map(one, [r["image_path"] for r in m], chunksize=200)

    def band(keep, lcf, ink):
        if keep < 0:
            return "err"
        if ink < 0.005:
            return "empty"
        if keep < 0.05:
            return "pure_noise"      # 丢弃
        if keep < 0.30:
            return "heavy_speckle"
        if keep < 0.60:
            return "light_speckle"
        return "clean"

    bands = collections.Counter()
    by_c = collections.defaultdict(collections.Counter)
    drop = []
    for r, (keep, lcf, ink) in zip(m, res):
        b = band(keep, lcf, ink)
        bands[b] += 1
        by_c[r["calligrapher"]][b] += 1
        if b == "pure_noise":
            drop.append(r["image_path"])

    tot = len(m)
    print()
    print("=== 全量分档 ===")
    for b in ["clean", "light_speckle", "heavy_speckle", "pure_noise", "empty", "err"]:
        if bands[b]:
            print(f"   {b:<15} {bands[b]:>7}  {100*bands[b]/tot:>6.2f}%")
    print(f"   {'合计':<15} {tot:>7}")
    print()
    print("=== 需要丢弃(纯噪点) 的书家 (>=3 张) ===")
    rows = sorted(by_c.items(), key=lambda kv: -kv[1]["pure_noise"])
    print(f"   {'书家':<10}{'总':>6}{'纯噪点':>8}{'占比':>8}{'重散点':>8}")
    for c, cnt in rows:
        if cnt["pure_noise"] >= 3:
            t = sum(cnt.values())
            print(f"   {c:<10}{t:>6}{cnt['pure_noise']:>8}"
                  f"{100*cnt['pure_noise']/t:>7.1f}%{cnt['heavy_speckle']:>8}")
    print()
    print(f"   合计丢弃 {len(drop)} 张")
    with open("_sync_work/drop_images.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(drop))
    print("   -> _sync_work/drop_images.txt")
    print()
    print("=== 重散点 (keep 0.05~0.30) 最多的书家 ===")
    rows2 = sorted(by_c.items(), key=lambda kv: -kv[1]["heavy_speckle"])
    for c, cnt in rows2[:12]:
        t = sum(cnt.values())
        print(f"   {c:<10}{t:>6}  重散点 {cnt['heavy_speckle']:>5} "
              f"({100*cnt['heavy_speckle']/t:>5.1f}%)")
