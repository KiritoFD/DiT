"""修复 v2 —— 顺序修正 + 逐步展示。

⚠ v1 的 bug: 先闭运算会把**散点连成网格**, 然后小孔填充把网格每格都填满
   -> 墨量爆炸 (实测 闭+填 = 6.49x, 渲染出来整张全黑)。
   正确顺序: ①去散点 -> ②闭运算(桥接断裂) -> ③小孔填充(描满)
"""
import csv, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import (label as cc_label, binary_closing, binary_opening,
                           binary_fill_holes)
from multiprocessing import Pool

os.chdir("/root/Workspace/xy/DiT")
ST = np.ones((3, 3), bool)
DROP = {"徐渭", "伊秉绶"}


def repair(g, min_cc=20, max_hole=50, do_close=True, do_fill=True):
    out = g
    # ① 去散点
    lab, nc = cc_label(out)
    if nc:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        out = np.isin(lab, np.where(sz >= min_cc)[0])
    # ② 闭运算
    if do_close:
        out = binary_closing(out, structure=ST)
    # ③ 只填小孔 (大封闭空间是字形本身的"白", 如口/日/回, 不能填)
    if do_fill:
        holes = binary_fill_holes(out) & ~out
        if holes.any():
            hl, hn = cc_label(holes)
            hs = np.bincount(hl.ravel()); hs[0] = 0
            out = out | np.isin(hl, np.where(hs < max_hole)[0])
    return out


def kratio(path):
    g = np.asarray(Image.open(path).convert("L")) < 128
    return float(binary_opening(g, structure=ST).sum() / max(g.sum(), 1))


if __name__ == "__main__":
    rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
    need = [r for r in rows if r["calligrapher"] not in DROP]
    with Pool(48) as p:
        ks = p.map(kratio, [r["image_path"] for r in need], chunksize=200)
    cand = [(r, k) for r, k in zip(need, ks) if 0.05 <= k < 0.90]
    print(f"排除 徐渭/伊秉绶 后 {len(need)} 张, keep 0.05~0.90 的 {len(cand)} 张")

    random.seed(9)
    samp = random.sample(cand, min(6, len(cand)))
    CELL, LBL = 256, 42
    TITLES = ["原始 GT", "①去散点", "①+②闭运算", "①+②+③小孔填充"]
    cv = Image.new("RGB", (4 * CELL + 5 * 10, len(samp) * (CELL + LBL) + 5 * 10),
                   (24, 24, 28))
    dr = ImageDraw.Draw(cv)
    fb = ImageFont.truetype("tools/fonts/simhei.ttf", 15)
    for i, (r, k) in enumerate(samp):
        y0 = 10 + i * (CELL + LBL + 10)
        g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
        s1 = repair(g, do_close=False, do_fill=False)
        s2 = repair(g, do_close=True, do_fill=False)
        s3 = repair(g, do_close=True, do_fill=True)
        for ci, im in enumerate([g, s1, s2, s3]):
            x0 = 10 + ci * (CELL + 10)
            arr = np.stack([np.where(im, 0, 255).astype(np.uint8)] * 3, -1)
            cv.paste(Image.fromarray(arr), (x0, y0 + LBL))
            if i == 0:
                dr.text((x0 + 4, y0 + 3), TITLES[ci], font=fb, fill=(160, 200, 255))
        dr.text((10, y0 + 24),
                "{}/{} keep={:.2f}  去散={:.2f} 闭={:.2f} 填={:.2f}".format(
                    r["calligrapher"], r["character"], k,
                    s1.sum() / max(g.sum(), 1), s2.sum() / max(g.sum(), 1),
                    s3.sum() / max(g.sum(), 1)),
                font=fb, fill=(255, 225, 140))
    cv.save("assets/repair_v2.png")
    print("-> assets/repair_v2.png")

    # 全量统计: 修复后 keep 提升多少
    def after(path):
        g = np.asarray(Image.open(path).convert("L")) < 128
        a = repair(g)
        return (binary_opening(a, structure=ST).sum() / max(a.sum(), 1),
                a.sum() / max(g.sum(), 1))
    with Pool(48) as p:
        res = p.map(after, [r["image_path"] for r, _ in cand], chunksize=100)
    nk = np.array([x[0] for x in res]); gr = np.array([x[1] for x in res])
    print(f"\n修复后: keep med={np.median(nk):.3f} (原 med="
          f"{np.median([k for _,k in cand]):.3f})   墨量变化 med={np.median(gr):.3f}")
    print(f"  修复后 keep>=0.90 的: {(nk>=0.90).mean()*100:.1f}%")
