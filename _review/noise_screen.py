"""筛查 GT 噪点图 + 试修法。

噪点定义 (针对"墨迹以外的散点/纹理"):
  N1 speckle_ink   = 小连通块(<20px)里的墨点 / 总墨点   -> 散点噪声
  N2 med3_diff     = |二值 - 3x3中值滤波(二值)| 的均值    -> 高频毛刺
  N3 small_cc_ratio= 小连通块数 / 总连通块数
修法候选:
  F1 形态学开运算 (3x3 腐蚀+膨胀)  -> 去散点, 保留主体笔画
  F2 连通块面积过滤 (<20px 丢掉)   -> 去孤立散点
  F1+F2 组合
"""
import csv, os, random, collections
import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label, median_filter, binary_opening

os.chdir("/root/Workspace/xy/DiT")
random.seed(3)
N = 1500

o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))


def metrics(gt):
    lab, nc = cc_label(gt)
    sizes = np.bincount(lab.ravel())[1:]
    small = sizes < 20
    ink = gt.sum()
    n1 = (sizes[small].sum() / ink) if ink else 0.0
    n3 = (small.sum() / len(sizes)) if len(sizes) else 0.0
    med = median_filter(gt.astype(np.uint8), size=3) > 0
    n2 = float((gt ^ med).mean())
    return n1, n2, n3, nc, ink


def load(p):
    return np.asarray(Image.open(p).convert("L").resize((256, 256), Image.LANCZOS)) < 128


def scan(rows, tag):
    rec = []
    for r in random.sample(rows, min(N, len(rows))):
        try:
            g = load(r["image_path"])
        except Exception:
            continue
        if not g.any():
            continue
        rec.append(metrics(g))
    a = np.array(rec)
    print(f"=== {tag} (n={len(a)}) ===")
    for i, nm in enumerate(["N1 散点墨比", "N2 中值毛刺", "N3 小碎块比"]):
        v = a[:, i]
        print(f"   {nm:<14} med={np.median(v):.4f}  p50={np.percentile(v,50):.4f}  "
              f"p90={np.percentile(v,90):.4f}  p99={np.percentile(v,99):.4f}")
    for thr in [0.05, 0.10, 0.15]:
        print(f"   N1 > {thr:.2f} 的占比: {(a[:,0]>thr).mean()*100:5.1f}%   "
              f"N2 > {thr:.3f} 的: {(a[:,1]>thr/10).mean()*100:5.1f}%")
    print()
    return a


ao = scan(o, "老数据")
aw = scan(w, "HCSU")

# ---- 联合判定: 任一指标高 = 噪点图 ----
def flag(a):
    return (a[:, 0] > 0.10) | (a[:, 1] > 0.015) | (a[:, 2] > 0.60)


print("=== 联合判定 (N1>0.10 或 N2>0.015 或 N3>0.60) ===")
fo, fw = flag(ao), flag(aw)
print(f"   老数据 噪点图 {fo.sum()}/{len(fo)} = {fo.mean()*100:.1f}%")
print(f"   HCSU  噪点图 {fw.sum()}/{len(fw)} = {fw.mean()*100:.1f}%")
print(f"   合并估计 约 {(fo.sum()+fw.sum())/(len(fo)+len(fw))*100:.1f}% "
      f"-> 全量 53,424 张里约 {int((fo.sum()+fw.sum())/(len(fo)+len(fw))*53424)} 张")
