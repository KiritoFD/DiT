"""直接 dump 那 8 张可疑图的像素统计。"""
import csv, os
import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
paths = set(open("_sync_work/too_ink.txt", encoding="utf-8").read().split("\n"))
sel = [r for r in rows if r["image_path"] in paths][:8]
for r in sel:
    im = Image.open(r["image_path"])
    a = np.asarray(im.convert("L"))
    u, c = np.unique(a, return_counts=True)
    top = sorted(zip(c, u), reverse=True)[:3]
    h, w = a.shape
    m = max(4, h // 32)
    edge = np.concatenate([a[:m].ravel(), a[-m:].ravel(),
                           a[:, :m].ravel(), a[:, -m:].ravel()])
    name = "{}/{}".format(r["calligrapher"], r["character"])
    print("{} {} {}: 暗于128={:.1f}%  最常见像素={}  边框均值={:.0f}  中心均值={:.0f}"
          .format(name, im.size, im.mode, 100 * (a < 128).mean(),
                  [(int(x[1]), int(x[0])) for x in top],
                  edge.mean(), a[h // 4:3 * h // 4, w // 4:3 * w // 4].mean()))
