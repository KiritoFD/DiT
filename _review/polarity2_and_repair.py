"""① 边框径向亮度剖面判极性(找黑框)  ② 修复对比: 闭运算+小孔填充(描满)"""
import csv, os, collections, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import (label as cc_label, binary_closing, binary_opening,
                           binary_fill_holes)
from multiprocessing import Pool

os.chdir("/root/Workspace/xy/DiT")
ST = np.ones((3, 3), bool)
DROP = {"徐渭", "伊秉绶"}


def profile(path):
    """从边框往内的平均亮度 (每 2px 一档, 共 8 档)"""
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    h, w = a.shape
    out = []
    for d in range(0, 16, 2):
        ring = np.concatenate([a[d, d:w - d], a[h - 1 - d, d:w - d],
                               a[d:h - d, d], a[d:h - d, w - 1 - d]])
        out.append(float(ring.mean()) / 255.0)
    return out


def kratio(path):
    g = np.asarray(Image.open(path).convert("L")) < 128
    return float(binary_opening(g, structure=ST).sum() / max(g.sum(), 1))


if __name__ == "__main__":
    rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))

    print("=== 边框径向亮度剖面 (从最外圈往内, 每 2px 一档) ===")
    print(f"   {'书家':<8}" + "".join(f"{d:>7}" for d in range(0, 16, 2)))
    for c in ["徐渭", "伊秉绶", "金农", "智永", "黄庭坚", "启功", "苏轼"]:
        sub = [r for r in rows if r["calligrapher"] == c][:120]
        if not sub:
            continue
        with Pool(48) as p:
            pr = p.map(profile, [r["image_path"] for r in sub], chunksize=20)
        m = np.array(pr).mean(0)
        print(f"   {c:<8}" + "".join(f"{v:>7.3f}" for v in m))

    # ---- 修复对比 ----
    need = [r for r in rows if r["calligrapher"] not in DROP]
    with Pool(48) as p:
        ks = p.map(kratio, [r["image_path"] for r in need], chunksize=200)
    cand = [(r, k) for r, k in zip(need, ks) if 0.05 <= k < 0.90]
    print(f"\n排除 徐渭/伊秉绶 后 {len(need)} 张, 其中 keep 0.05~0.90 = {len(cand)} 张")

    random.seed(9)
    samp = random.sample(cand, min(6, len(cand)))
    CELL, LBL = 256, 42
    cv = Image.new("RGB", (4 * CELL + 5 * 10, len(samp) * (CELL + LBL) + 5 * 10),
                   (24, 24, 28))
    dr = ImageDraw.Draw(cv)
    fb = ImageFont.truetype("tools/fonts/simhei.ttf", 15)
    TITLES = ["原始 GT", "闭运算(描满)", "闭+小孔填充", "闭+填+去散点(<20px)"]
    for i, (r, k) in enumerate(samp):
        y0 = 10 + i * (CELL + LBL + 10)
        g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
        a = binary_closing(g, structure=ST)
        b = a.copy()
        holes = binary_fill_holes(b) & ~b
        if holes.any():
            hl, hn = cc_label(holes)
            hs = np.bincount(hl.ravel()); hs[0] = 0
            b = b | np.isin(hl, np.where(hs < 50)[0])
        c2 = b.copy()
        lab, nc = cc_label(c2)
        if nc:
            sz = np.bincount(lab.ravel()); sz[0] = 0
            c2 = np.isin(lab, np.where(sz >= 20)[0])
        for ci, im in enumerate([g, a, b, c2]):
            x0 = 10 + ci * (CELL + 10)
            arr = np.stack([np.where(im, 0, 255).astype(np.uint8)] * 3, -1)
            cv.paste(Image.fromarray(arr), (x0, y0 + LBL))
            if i == 0:
                dr.text((x0 + 4, y0 + 3), TITLES[ci], font=fb, fill=(160, 200, 255))
        dr.text((10, y0 + 24),
                "{}/{}  keep={:.2f}  闭={:.2f} 闭+填={:.2f} 闭+填+去散={:.2f}".format(
                    r["calligrapher"], r["character"], k,
                    a.sum() / max(g.sum(), 1), b.sum() / max(g.sum(), 1),
                    c2.sum() / max(g.sum(), 1)),
                font=fb, fill=(255, 225, 140))
    cv.save("assets/repair_cmp.png")
    print("-> assets/repair_cmp.png")
