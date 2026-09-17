"""建 data/50k/: 把清洗后保留的图 + std 全部复制过去, 生成 assets/train_50k.csv。

分类口径:
  drop          宋高宗整批 + keep<0.60            -> **不复制**
  需要后处理     mode=L 且 keep<0.90              -> 有纹理, 可考虑清理
  干净可用       mode=RGB, 或 mode=L 且 keep>=0.90
"""
import csv, os, shutil, collections, random
import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening
from multiprocessing import Pool

os.chdir("/root/Workspace/xy/DiT")
ST = np.ones((3, 3), bool)
SRC_CSV = "assets/train_merged_clean.csv"
DST = "data/50k"


def info(path):
    """返回 (mode, keep)"""
    try:
        im = Image.open(path)
        md = im.mode
        g = np.asarray(im.convert("L")) < 128
    except Exception:
        return ("ERR", -1.0)
    ink = max(int(g.sum()), 1)
    if ink < 10:
        return (md, 0.0)
    return (md, float(binary_opening(g, structure=ST).sum() / ink))


def cp(pair):
    s, d = pair
    try:
        shutil.copy2(s, d)
        return 1
    except Exception:
        return 0


if __name__ == "__main__":
    rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))
    print(f"保留集 {len(rows)} 行, 开始分析 ...", flush=True)
    with Pool(48) as p:
        infos = p.map(info, [r["image_path"] for r in rows], chunksize=200)

    cat = collections.Counter()
    need_post, clean = [], []
    for r, (md, k) in zip(rows, infos):
        if md == "RGB" or k >= 0.90:
            cat["干净可用"] += 1
            clean.append((r, md, k))
        else:
            cat["需要后处理"] += 1
            need_post.append((r, md, k))
    print()
    print("=== 保留集内部构成 ===")
    print(f"   干净可用    {cat['干净可用']:>6}  ({100*cat['干净可用']/len(rows):.2f}%)")
    print(f"   需要后处理  {cat['需要后处理']:>6}  ({100*cat['需要后处理']/len(rows):.2f}%)")
    mdd = collections.Counter(md for _, md, _ in need_post)
    print(f"      其中 mode 分布: {dict(mdd)}")
    kk = np.array([k for _, _, k in need_post])
    if len(kk):
        print(f"      keep: med={np.median(kk):.3f} p10={np.percentile(kk,10):.3f} "
              f"min={kk.min():.3f}")

    # ---- 复制 ----
    os.makedirs(f"{DST}/imgs", exist_ok=True)
    os.makedirs(f"{DST}/std", exist_ok=True)
    pairs, new_rows = [], []
    for i, r in enumerate(rows):
        nid = f"{i:06d}"
        di, ds = f"{DST}/imgs/{nid}.png", f"{DST}/std/{nid}.png"
        pairs.append((r["image_path"], di))
        pairs.append((r["std_path"], ds))
        nr = dict(r)
        nr["image_path"] = di
        nr["std_path"] = ds
        nr["src_image_path"] = r["image_path"]
        nr["old_50k_id"] = nid
        new_rows.append(nr)
    print(f"\n复制 {len(pairs)} 个文件 -> {DST}/ ...", flush=True)
    with Pool(48) as p:
        ok = p.map(cp, pairs, chunksize=100)
    print(f"   成功 {sum(ok)}/{len(pairs)}")

    # ---- 写 CSV ----
    cols = list(new_rows[0].keys())
    with open("assets/train_50k.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(new_rows)
    print("-> assets/train_50k.csv")
    print(f"   保留后: 张 {len(new_rows)}  书家 "
          f"{len({r['calligrapher'] for r in new_rows})}  字 "
          f"{len({r['character'] for r in new_rows})}  (书体,字) "
          f"{len({(r['script'], r['character']) for r in new_rows})}")

    # ---- 需要后处理 的抽样海报 ----
    random.seed(4)
    samp = random.sample(need_post, min(16, len(need_post)))
    CELL, LBL, NC = 256, 40, 4
    NR = (len(samp) + NC - 1) // NC
    cv = Image.new("RGB", (NC * CELL + (NC + 1) * 8, NR * (CELL + LBL) + (NR + 1) * 8),
                   (24, 24, 28))
    from PIL import ImageDraw, ImageFont
    dr = ImageDraw.Draw(cv)
    fb = ImageFont.truetype("tools/fonts/simhei.ttf", 14)
    for i, (r, md, k) in enumerate(samp):
        rr, cc = i // NC, i % NC
        x0, y0 = 8 + cc * (CELL + 8), 8 + rr * (CELL + LBL + 8)
        g = np.asarray(Image.open(r["image_path"]).convert("L")
                       .resize((CELL, CELL), Image.LANCZOS)) < 128
        cv.paste(Image.fromarray(np.stack([np.where(g, 0, 255).astype(np.uint8)] * 3, -1)),
                 (x0, y0 + LBL))
        dr.text((x0 + 4, y0 + 3), f"{r['calligrapher']}/{r['character']} "
                                 f"{md} keep={k:.2f}", font=fb, fill=(255, 225, 140))
    cv.save("assets/50k_need_post.png")
    print("-> assets/50k_need_post.png")
