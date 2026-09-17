"""数据质量量化: 骨架对齐 + GT 噪声/碎片度。"""
import csv, os, random, collections
import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label

os.chdir("/root/Workspace/xy/DiT")
random.seed(11)
N = 800

o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))


def stat(rows, tag):
    rec = []
    for r in random.sample(rows, min(N, len(rows))):
        try:
            gt = np.asarray(Image.open(r["image_path"]).convert("L")
                            .resize((256, 256), Image.LANCZOS)) < 128
            st = np.asarray(Image.open(r["std_path"]).convert("L")
                            .resize((256, 256), Image.NEAREST)) < 128
        except Exception:
            continue
        if not st.any() or not gt.any():
            continue
        # ① 骨架落墨率: std 骨架像素有多少落在 GT 墨内 (对齐质量)
        hit = (st & gt).sum() / st.sum()
        # ② GT 墨有多少被骨架覆盖 (骨架完整性)
        cov = (st & gt).sum() / gt.sum()
        # ③ GT 碎片度: 连通块数 / 墨面积 (噪声高 -> 大量小碎块)
        lab, nc = cc_label(gt)
        # 去掉极小碎块后的主要连通块占比
        sizes = np.bincount(lab.ravel())[1:]
        main = sizes.max() / gt.sum() if len(sizes) else 0
        small = (sizes < 20).sum() / max(len(sizes), 1)
        rec.append((hit, cov, nc, main, small, gt.mean()))
    a = np.array(rec)
    print(f"=== {tag}  (n={len(a)}) ===")
    def pr(name, i, pct=True):
        v = a[:, i]
        print(f"   {name:<22} med={np.median(v):.4f}  p10={np.percentile(v,10):.4f}  "
              f"p90={np.percentile(v,90):.4f}")
    pr("① 骨架落墨率 (对齐)", 0)
    pr("② GT 墨被骨架覆盖", 1)
    pr("③ GT 连通块数", 2)
    pr("④ 最大连通块占墨比", 3)
    pr("⑤ 碎块(<20px)占比", 4)
    pr("⑥ GT 墨点率", 5)
    # 差样本比例
    print(f"   落墨率 <0.90 的: {(a[:,0]<0.90).mean()*100:.1f}%")
    print(f"   落墨率 <0.75 的: {(a[:,0]<0.75).mean()*100:.1f}%")
    print(f"   最大连通块 <0.5 的(碎片化严重): {(a[:,3]<0.5).mean()*100:.1f}%")
    print()
    return a


ao = stat(o, "老数据")
aw = stat(w, "HCSU")
