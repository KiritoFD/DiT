"""用 Chamfer 距离定 std 的最佳缩放 (落墨率有"越小分越高"的偏, 不可用)。

chamfer(std->GT) = 每个 std 像素到最近 GT 墨像素的平均距离 (px)
  太大 -> std 伸出 GT 外或错位
太小/无差别 -> std 缩在 GT 粗笔画内 (也会很小!)
所以同时报 **双向**: chamfer(std->GT) 与 chamfer(GT->std)
  前者大 = std 超出; 后者大 = std 覆盖不足
"""
import csv, os, random
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

os.chdir("/root/Workspace/xy/DiT")
random.seed(21)


def bbox(m):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def place(crop, tw, th, cx=128, cy=128):
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (max(tw, 1), max(th, 1)), Image.NEAREST)
    a = np.asarray(im) > 127
    out = np.zeros((256, 256), bool)
    x0 = int(round(cx - tw / 2)); y0 = int(round(cy - th / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    h = min(a.shape[0] - sy0, 256 - y0); w = min(a.shape[1] - sx0, 256 - x0)
    if h > 0 and w > 0:
        out[y0:y0 + h, x0:x0 + w] = a[sy0:sy0 + h, sx0:sx0 + w]
    return out


def norm(s, target):
    b = bbox(s)
    c = s[b[1]:b[3] + 1, b[0]:b[2] + 1]
    h, w = c.shape
    sc = float(target) / max(h, w)
    return place(c, max(int(round(w * sc)), 1), max(int(round(h * sc)), 1))


def cham(s, g):
    """(std->GT 平均距离, GT->std 平均距离)"""
    dt_g = distance_transform_edt(~g)
    dt_s = distance_transform_edt(~s)
    a = dt_g[s].mean() if s.any() else np.nan
    b = dt_s[g].mean() if g.any() else np.nan
    return a, b


rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
samp = random.sample(rows, 500)
cfgs = {"原样": None, "缩到192": 192, "缩到208": 208, "缩到224": 224,
        "缩到240": 240, "缩到256": 256}
acc = {k: [] for k in cfgs}
for r in samp:
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    if not g.any() or not s.any():
        continue
    for k, t in cfgs.items():
        acc[k].append(cham(s if t is None else norm(s, t), g))

print(f"n={len(acc['原样'])}")
print(f"{'方案':<10}{'std->GT(超出)':>15}{'GT->std(覆盖不足)':>19}{'合计':>9}")
for k in cfgs:
    a = np.array(acc[k])
    ma, mb = np.nanmedian(a[:, 0]), np.nanmedian(a[:, 1])
    print(f"   {k:<10}{ma:>12.2f}px{mb:>16.2f}px{ma+mb:>9.2f}")
