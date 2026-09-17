"""测 GT / std 的墨迹 bbox 尺寸, 看 std 到底小多少。"""
import csv, os, random, collections
import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
random.seed(21)


def bbox(m):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
samp = random.sample(rows, 600)

gw, gh, sw, sh, ratio = [], [], [], [], []
gcy, gcx, scy, scx = [], [], [], []
for r in samp:
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    bg, bs = bbox(g), bbox(s)
    if bg is None or bs is None:
        continue
    gw.append(bg[2] - bg[0]); gh.append(bg[3] - bg[1])
    sw.append(bs[2] - bs[0]); sh.append(bs[3] - bs[1])
    ratio.append(max(sw[-1], sh[-1]) / max(gw[-1], gh[-1]))
    gcx.append((bg[0] + bg[2]) / 2); gcy.append((bg[1] + bg[3]) / 2)
    scx.append((bs[0] + bs[2]) / 2); scy.append((bs[1] + bs[3]) / 2)

gw, gh, sw, sh = map(np.array, (gw, gh, sw, sh))
print(f"n={len(gw)}")
print(f"  GT  墨迹 bbox: 宽 med={np.median(gw):.0f} (p10 {np.percentile(gw,10):.0f} "
      f"p90 {np.percentile(gw,90):.0f})   高 med={np.median(gh):.0f}")
print(f"  std 墨迹 bbox: 宽 med={np.median(sw):.0f} (p10 {np.percentile(sw,10):.0f} "
      f"p90 {np.percentile(sw,90):.0f})   高 med={np.median(sh):.0f}")
print(f"  尺寸比 std/GT: med={np.median(ratio):.3f}  p10={np.percentile(ratio,10):.3f} "
      f"p90={np.percentile(ratio,90):.3f}")
print(f"  中心偏移: dx med={np.median(np.array(scx)-np.array(gcx)):+.1f} "
      f"dy med={np.median(np.array(scy)-np.array(gcy)):+.1f}")
print(f"  GT 填充率: 宽 med={np.median(gw)/256:.3f}  高 med={np.median(gh)/256:.3f}")
print(f"  std 填充率: 宽 med={np.median(sw)/256:.3f}  高 med={np.median(sh)/256:.3f}")
