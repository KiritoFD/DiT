"""最终: 丢 徐渭/伊秉绶 -> 对 keep<0.90 的图做「描满」修复 -> 重建 50k。

修复配方 (v2 定稿):
  ① 去散点: 丢掉 < 20px 的孤立连通块
  ② 3x3 闭运算: 桥接笔画断裂, 让笔画变实
  ✗ 不做 binary_fill_holes —— 它会把"闭运算连成的网格"逐格填满, 墨量爆炸
    (实测 17.86x / 6.49x / 5.66x, 渲染出来整张全黑)
"""
import csv, os, random, shutil, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label as cc_label, binary_closing, binary_opening
from multiprocessing import Pool

os.chdir("/root/Workspace/xy/DiT")
ST = np.ones((3, 3), bool)
DROP = {"徐渭", "伊秉绶"}
THR = 0.90


def proc(r):
    p = r["image_path"]
    g = np.asarray(Image.open(p).convert("L")) < 128
    ink0 = max(int(g.sum()), 1)
    keep0 = binary_opening(g, structure=ST).sum() / ink0
    if keep0 >= THR:
        return (p, keep0, keep0, 1.0, 0)          # 已干净, 不动
    out = g
    lab, nc = cc_label(out)
    if nc:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        out = np.isin(lab, np.where(sz >= 20)[0])
    out = binary_closing(out, structure=ST)
    keep1 = binary_opening(out, structure=ST).sum() / max(out.sum(), 1)
    Image.fromarray(np.where(out, 0, 255).astype(np.uint8)).save(p)
    return (p, keep0, keep1, out.sum() / ink0, 1)


if __name__ == "__main__":
    rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
    kept = [r for r in rows if r["calligrapher"] not in DROP]
    print(f"丢 {'/'.join(DROP)}: {len(rows)} -> {len(kept)}  (丢 {len(rows)-len(kept)})")

    print("修复中 (keep<{:.2f} 的图)...".format(THR), flush=True)
    with Pool(48) as p:
        res = p.map(proc, kept, chunksize=100)
    fixed = [x for x in res if x[4] == 1]
    k0 = np.array([x[1] for x in fixed]); k1 = np.array([x[2] for x in fixed])
    gr = np.array([x[3] for x in fixed])
    print(f"   修复 {len(fixed)} 张")
    if len(fixed):
        print(f"   keep: {np.median(k0):.3f} -> {np.median(k1):.3f}   "
              f"修复后 >=0.90 的 {100*(k1>=0.90).mean():.1f}%")
        print(f"   墨量变化 med={np.median(gr):.3f} p90={np.percentile(gr,90):.3f} "
              f"max={gr.max():.2f}")

    # 重建 CSV
    with open("assets/train_50k.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
        w.writeheader(); w.writerows(kept)
    print(f"\n-> assets/train_50k.csv  {len(kept)} 行")
    print(f"   书家 {len({r['calligrapher'] for r in kept})}  "
          f"字 {len({r['character'] for r in kept})}  "
          f"(书体,字) {len({(r['script'], r['character']) for r in kept})}")
    print("   来源:", dict(collections.Counter(r["source"] for r in kept)))

    # 修复前后海报
    random.seed(11)
    samp = random.sample(fixed, min(6, len(fixed))) if fixed else []
    if samp:
        CELL, LBL = 256, 42
        cv = Image.new("RGB", (2 * CELL + 3 * 10, len(samp) * (CELL + LBL) + 5 * 10),
                       (24, 24, 28))
        dr = ImageDraw.Draw(cv)
        fb = ImageFont.truetype("tools/fonts/simhei.ttf", 15)
        for i, (p, a, b, g_, f_) in enumerate(samp):
            y0 = 10 + i * (CELL + LBL + 10)
            src = [r for r in rows if r["image_path"] == p][0]
            # 原始图在源目录里还在 (data/50k 是副本)
            o = np.asarray(Image.open(src["src_image_path"]).convert("L")) < 128
            n = np.asarray(Image.open(p).convert("L")) < 128
            for ci, im in enumerate([o, n]):
                x0 = 10 + ci * (CELL + 10)
                arr = np.stack([np.where(im, 0, 255).astype(np.uint8)] * 3, -1)
                cv.paste(Image.fromarray(arr), (x0, y0 + LBL))
                if i == 0:
                    dr.text((x0 + 4, y0 + 3), ["修复前", "修复后"][ci],
                            font=fb, fill=(160, 200, 255))
            dr.text((10, y0 + 24),
                    "{}/{}  keep {:.2f}->{:.2f}".format(
                        src["calligrapher"], src["character"], a, b),
                    font=fb, fill=(255, 225, 140))
        cv.save("assets/repair_final.png")
        print("-> assets/repair_final.png")
