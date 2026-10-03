"""定 std 缩放: ①按 GT/std 的 bbox 长边均值算比例  ②试 90%(230) 等目标值。
指标: Chamfer 双向 (落墨率有"越小分越高"的偏, 不能用)。
"""
import csv, os, random, sys
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

os.chdir("/root/Workspace/xy/DiT")
random.seed(21)


def bbox(m):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def place(crop, tw, th):
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (max(tw, 1), max(th, 1)), Image.NEAREST)
    a = np.asarray(im) > 127
    out = np.zeros((256, 256), bool)
    x0 = int(round(128 - tw / 2)); y0 = int(round(128 - th / 2))
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
    return place(c, int(round(w * sc)), int(round(h * sc)))


rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
samp = random.sample(rows, 400)

gl, sl = [], []
for r in samp:
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    bg, bs = bbox(g), bbox(s)
    if bg and bs:
        gl.append(max(bg[2] - bg[0], bg[3] - bg[1]) + 1)
        sl.append(max(bs[2] - bs[0], bs[3] - bs[1]) + 1)
gl, sl = np.array(gl), np.array(sl)
print(f"n={len(gl)}")
print(f"  GT  长边: mean={gl.mean():.1f} med={np.median(gl):.1f}")
print(f"  std 长边: mean={sl.mean():.1f} med={np.median(sl):.1f}")
print(f"  **比例 GT/std: mean 比 {gl.mean()/sl.mean():.4f}   "
      f"中位比 {np.median(gl)/np.median(sl):.4f}**")
print(f"  -> 目标长边 = std 长边 x 比例 = {np.median(sl)*gl.mean()/sl.mean():.1f}px")

TGTS = [None, 208, 224, 230, 240, 256]
acc = {t: [] for t in TGTS}
for r in samp:
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    if not g.any() or not s.any():
        continue
    dg, ds = distance_transform_edt(~g), distance_transform_edt(~s)
    for t in TGTS:
        sv = s if t is None else norm(s, t)
        acc[t].append((dg[sv].mean(), ds[g].mean()))

print()
print(f"{'目标长边':<10}{'std->GT':>10}{'GT->std':>10}{'合计':>9}")
for t in TGTS:
    a = np.array(acc[t])
    ma, mb = np.nanmedian(a[:, 0]), np.nanmedian(a[:, 1])
    name = "原样" if t is None else str(t)
    print(f"   {name:<10}{ma:>9.2f}{mb:>10.2f}{ma+mb:>9.2f}")
