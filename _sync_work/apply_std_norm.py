"""把 data/50k/std/ 全部归一化: 裁到 bbox -> 长边缩到 230(保持比例) -> 居中。
缩放后**重新骨架化+3x3膨胀**, 保住 3px 线宽属性。

目标 230 的依据: GT 长边 med=228.5, std 原长边 med=172 -> 比例 1.33 -> 229px
"""
import csv, os, random
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize
from multiprocessing import Pool

os.chdir("/root/Workspace/xy/DiT")
ST = np.ones((3, 3), bool)
TARGET = 230


def bbox(m):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def one(path):
    a = np.asarray(Image.open(path).convert("L"))
    s = a < 128
    b = bbox(s)
    if b is None:
        return (path, 0, 0)
    c = s[b[1]:b[3] + 1, b[0]:b[2] + 1]
    h, w = c.shape
    sc = float(TARGET) / max(h, w)
    tw, th = max(int(round(w * sc)), 1), max(int(round(h * sc)), 1)
    im = Image.fromarray((c * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)
    big = np.asarray(im) > 127
    # 缩放后线宽变了 -> 重新骨架化再膨胀回 3px
    sk = skeletonize(big)
    if not sk.any():
        sk = big
    big3 = binary_dilation(sk, structure=ST)
    out = np.zeros((256, 256), bool)
    x0, y0 = int(round(128 - tw / 2)), int(round(128 - th / 2))
    hh = min(big3.shape[0], 256 - y0)
    ww = min(big3.shape[1], 256 - x0)
    out[y0:y0 + hh, x0:x0 + ww] = big3[:hh, :ww]
    Image.fromarray(np.where(out, 0, 255).astype(np.uint8)).save(path)
    return (path, max(w, h), max(tw, th))


if __name__ == "__main__":
    rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
    paths = [r["std_path"] for r in rows]
    print(f"归一化 {len(paths)} 个 std ...", flush=True)
    with Pool(48) as p:
        res = p.map(one, paths, chunksize=200)
    ok = [x for x in res if x[1] > 0]
    print(f"   处理 {len(ok)} 个")

    # 核对
    gl, sl, lw = [], [], []
    for r in random.sample(rows, 400):
        g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
        s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
        bg, bs = bbox(g), bbox(s)
        if bg and bs:
            gl.append(max(bg[2] - bg[0], bg[3] - bg[1]) + 1)
            sl.append(max(bs[2] - bs[0], bs[3] - bs[1]) + 1)
            sk = skeletonize(s)
            if sk.any():
                lw.append(s.sum() / sk.sum())
    print(f"   归一化后 std 长边 med={np.median(sl):.1f}  "
          f"GT 长边 med={np.median(gl):.1f}  比值={np.median(sl)/np.median(gl):.3f}")
    print(f"   线宽 med={np.median(lw):.2f} (目标 ~3.5, 与 3x3 膨胀一致)")
