"""1) 用骨架叠加客观判极性  2) 丢 徐渭/伊秉绶  3) 改进修复: 描满(闭运算+小孔填充)

修复思路修正:
  旧: 3x3 **开运算** -> 去掉散点, 但把细笔画削碎  ✗ 方向错了
  新: ① 小碎块过滤(<20px)  ② **闭运算**桥接笔画断裂  ③ **小孔填充**(只填 <50px 的孔,
      不填字形本身的大封闭空间如「口」「日」)  -> 描满
"""
import csv, os, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import (label as cc_label, binary_closing, binary_opening,
                           binary_fill_holes)
from multiprocessing import Pool

os.chdir("/root/Workspace/xy/DiT")
ST = np.ones((3, 3), bool)
DROP_CALLIG = {"徐渭", "伊秉绶"}


def repair(g):
    """描满: 去散点 -> 闭运算 -> 只填小孔"""
    # ① 去掉 < 20px 的孤立碎块
    lab, nc = cc_label(g)
    if nc:
        sz = np.bincount(lab.ravel())
        sz[0] = 0
        g = np.isin(lab, np.where(sz >= 20)[0])
    # ② 闭运算: 桥接笔画内 1~2px 的断裂
    g = binary_closing(g, structure=ST)
    # ③ 只填**小孔** (<50px): 大封闭空间是字形本身的"白"(口/日/回), 不能填
    holes = binary_fill_holes(g) & ~g
    if holes.any():
        hl, hn = cc_label(holes)
        if hn:
            hs = np.bincount(hl.ravel())
            hs[0] = 0
            small = np.where(hs < 50)[0]
            g = g | np.isin(hl, small)
    return g


def analyze(path):
    g = np.asarray(Image.open(path).convert("L")) < 128
    s = np.asarray(Image.open(path.replace("/imgs/", "/std/")).convert("L")) < 128
    hit = (s & g).sum() / max(s.sum(), 1)      # 骨架落在暗像素的比例
    ink = g.sum() / g.size
    keep_op = binary_opening(g, structure=ST).sum() / max(g.sum(), 1)
    return hit, ink, keep_op


if __name__ == "__main__":
    rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))

    # ---- 1. 极性: 骨架落暗率 ----
    print("=== 极性判定 (骨架落暗率; <0.5 说明该图暗像素不是墨) ===")
    for c in ["徐渭", "伊秉绶", "金农", "赵之谦", "黄庭坚", "智永", "启功"]:
        sub = [r for r in rows if r["calligrapher"] == c]
        if not sub:
            continue
        with Pool(48) as p:
            res = p.map(analyze, [r["image_path"] for r in sub[:300]], chunksize=50)
        hits = np.array([x[0] for x in res])
        inks = np.array([x[1] for x in res])
        frac = (hits < 0.5).mean()
        print(f"   {c:<8} n={len(sub):>5}  落暗率 med={np.median(hits):.3f}  "
              f"墨点率 med={np.median(inks):.3f}  疑似反色占比={100*frac:5.1f}%")
    print()

    # ---- 2. 需要后处理的图: 修复前后对比 ----
    need = [r for r in rows if r["calligrapher"] not in DROP_CALLIG]
    print(f"排除 {'/'.join(DROP_CALLIG)} 后 {len(need)} 张")

    def post(r):
        g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
        k = binary_opening(g, structure=ST).sum() / max(g.sum(), 1)
        return (r, k, g)

    with Pool(48) as p:
        got = p.map(lambda r: (r, binary_opening(
            np.asarray(Image.open(r["image_path"]).convert("L")) < 128,
            structure=ST).sum() / max((np.asarray(Image.open(r["image_path"])
                                                      .convert("L")) < 128).sum(), 1)),
                    need, chunksize=200)
    need_post = [(r, k) for r, k in got if 0.05 <= k < 0.90]
    print(f"需要后处理 (keep 0.05~0.90): {len(need_post)} 张")

    import random
    random.seed(9)
    samp = random.sample(need_post, min(8, len(need_post)))
    CELL, LBL = 256, 42
    cv = Image.new("RGB", (4 * CELL + 5 * 10, len(samp) * (CELL + LBL) + 5 * 10),
                   (24, 24, 28))
    dr = ImageDraw.Draw(cv)
    fb = ImageFont.truetype("tools/fonts/simhei.ttf", 15)
    TITLES = ["原始", "闭运算+小孔填充(描满)", "闭+填+去散点", "开运算(旧, 反了)"]
    stats = []
    for i, (r, k) in enumerate(samp):
        y0 = 10 + i * (CELL + LBL + 10)
        g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
        a = binary_closing(g, structure=ST)
        holes = binary_fill_holes(a) & ~a
        if holes.any():
            hl, hn = cc_label(holes)
            hs = np.bincount(hl.ravel()); hs[0] = 0
            a = a | np.isin(hl, np.where(hs < 50)[0])
        b = repair(g)
        c = binary_opening(g, structure=ST)
        for ci, im in enumerate([g, a, b, c]):
            x0 = 10 + ci * (CELL + 10)
            arr = np.stack([np.where(im, 0, 255).astype(np.uint8)] * 3, -1)
            cv.paste(Image.fromarray(arr), (x0, y0 + LBL))
            if i == 0:
                dr.text((x0 + 4, y0 + 3), TITLES[ci], font=fb, fill=(160, 200, 255))
        dr.text((10, y0 + 24),
                f"{r['calligrapher']}/{r['character']}  keep={k:.2f}  "
                f"闭+填保留={a.sum()/max(g.sum(),1):.2f}  去散点后={b.sum()/max(g.sum(),1):.2f}",
                font=fb, fill=(255, 225, 140))
        stats.append((a.sum() / max(g.sum(), 1), b.sum() / max(g.sum(), 1)))
    cv.save("assets/repair_cmp.png")
    print("-> assets/repair_cmp.png")
    sa = np.array(stats)
    print(f"修复保留率: 闭+填 med={np.median(sa[:,0]):.3f}  闭+填+去散点 "
          f"med={np.median(sa[:,1]):.3f}")
